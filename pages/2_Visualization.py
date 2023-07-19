import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from engine.Engine import Engine
from utils.utils import call_arviz_lib
from utils.chat_help import *
from utils.visual_plots import *
from utils.visual_page import *
import pandas as pd
    
def run():
    # Setting page title and header
    st.set_page_config(page_title="BN Results", page_icon=":robot_face:", layout='wide')  # , layout="wide"
    st.title("Bayesian Network Results")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    
    new_dict = get_node_fullname()

    # Sidebar to create
    st.sidebar.title('Select Variable')
    window = st.sidebar.slider('Num Samples', min_value=1000, max_value=100000, step=500)
    node_value = st.sidebar.selectbox('Node', list(new_dict.values()))
    node_key = find_key(new_dict, node_value)
    for i in ['state_names','probs']:
        if i not in st.session_state:
            st.session_state[i] = []

    # -------------------
    # NETWORK PLOT
    st.divider()
    intro = st.container()
    with intro:
        Network()
    
    # -------------------
    # CHECK PROBABILITIES
    st.divider()
    st.markdown("""### Node: {}""".format(node_value))

    # check historical data to enter states/probs
    with open('./engine/node_dic.json', 'r') as file:
        node_dic = json.load(file)
        node_value_lower = node_value.lower()
        conditions = [node_dic[node_value],
                        any(keyword in node_value_lower 
                            for keyword in ['buy stradd', 'buying stradd',
                                            'market correc', 'investment loss',
                                            'sell futu', 'portfolio loss',
                                            'short sell','buying str','selling fut']),
                    ]

    short_name, child_node = get_child_node(node_value)
    
    if 'prior' not in st.session_state:
        st.session_state.prior = {}
    if 'condi' not in st.session_state:
        st.session_state.condi = {}


    if any(conditions): # have historical data
        st.markdown('**We used historical data for this node.**')
        # Conditional probs table
        cpds = engine.BN_model.get_cpds() 
        
        # Get Conditional Probability for a specific node
        for cpd in cpds:
            if cpd.variable == node_key:
                cpd_table = cpd_to_df(cpd)
                st.dataframe(cpd_table)

    else: # no historical data
        st.markdown('**We don\'t have historical data for this node. Change the state & probabilities to see how the result changes.**')
        prior, cond = st.columns([5,10])

        with prior:
            st.markdown('**Prior Probs**')
            number = st.slider('Number of states', 1, 5)
            num_rows = pd.to_numeric(number)
            prior_prob = prior_prob_table(node_value, short_name, 
                                        num_rows)

            if isinstance(prior_prob, pd.DataFrame):
                st.session_state.prior = prior_prob.to_json()

        with cond:
            st.markdown(f'**Conditional Probs with {child_node}**')
        
            if isinstance(prior_prob, pd.DataFrame) and 'State' in prior_prob.columns:
                if prior_prob['State'].count().sum() != number:
                    st.warning('Enter Prior first')
                else:
                    condi_prob = cond_prob_table(prior_prob,
                                                pd.to_numeric(number),
                                                short_name, child_node)
                if isinstance(condi_prob, pd.DataFrame):        
                    st.session_state.condi = condi_prob.to_json()


    # --------------------------
    # FINAL PLOTS
    st.divider()
    visuals = st.container()
    with visuals: # for a specific node
        pred = get_sim_y(node_key, window)
        Posterior(pred)
        Forest(pred)
        Summary(pred)

def main():
    run()

if __name__ == '__main__':
    main()