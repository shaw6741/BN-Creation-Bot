import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from engine.Engine import Engine
from utils.utils import call_arviz_lib
from utils.chat_help import *
import pandas as pd



engine = Engine()
engine.start()

def get_sim_y(node, n_samples):
    predict_ = engine.BN_model.model_simulate(n_samples)
    predict_ = predict_[node]
    return predict_.values

def Network():
    st.markdown('#### Network Plot')
    fig = engine.BN_model.plot_networkx()
    st.write(fig)

def Distribution(predict_y):
    st.markdown('#### Distribution')
    obj = call_arviz_lib().get_dist(predict_y)
    st.pyplot(obj)

def Posterior(predict_y):
    st.markdown('#### Posterior Distribution')
    st.pyplot(call_arviz_lib().get_posterior(predict_y))

def Forest(predict_y):
    st.markdown('#### Plot Forest')
    st.pyplot(call_arviz_lib().get_plot_forest(predict_y))

def Summary(predict_y):
    st.markdown('#### Summary Table')
    df = pd.DataFrame(call_arviz_lib().get_summary(predict_y))  # st.dataframe
    st.write(df)

def cpd_to_df(cpd):
    variables = cpd.variables
    cardinality = cpd.cardinality
    values = cpd.values.flatten()

    # Generate all combinations of variable states
    index_tuples = pd.MultiIndex.from_product([range(card) for card in cardinality], names=variables)
    df = pd.DataFrame({'Probabilities': values}, index=index_tuples)
    # Rename columns
    for var in variables:
        state_names = cpd.state_names[var]
        state_names = np.array(state_names).astype(str)
        column_name = f'{var} ({", ".join(state_names)})'
        df.rename(columns={var: column_name}, inplace=True)

    # Sort columns
    df = df.reorder_levels(variables, axis=0)
    df.sort_index(axis=0, inplace=True)

    return df

    
def run():
    # Setting page title and header
    st.set_page_config(page_title="BN Results", page_icon=":robot_face:")  # , layout="wide"
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

    st.divider()
    intro = st.container()
    with intro:
        Network()
        
    st.divider()
    body = st.container()
    with body: # for a specific node
        st.markdown("""### Node: {}""".format(node_value))
        
        # check historical data, enter states/probs
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

        if any(conditions):
            st.markdown('**We used historical data for this node.**')
        else:
            st.markdown('**We don\'t have historical data for this node. Change the state & probabilities to see how the result changes.**')
            short_name, child_node = get_child_node(node_value)
            
            number = st.slider('Number of states', 1, 5)

            for i in range(number):
                state_num = i+1
                left, right = st.columns(2)
                state_names, probs = [], []
                with left:
                    state_name = st.text_input(f'State {state_num} Name:')
                    state_names.append(state_name)
                    
                with right:
                    prob = st.slider(f'Probability for State {state_num}', 0.00, 1.00, 0.05)
                    probs.append(prob)

            st.session_state.state_names.append(state_names)
            st.session_state.probs.append(probs)
            st.write(st.session_state.state_names)
            st.write(st.session_state.probs)
            
        # Conditiona 
        cpds = engine.BN_model.get_cpds() 
        
        # Get Conditional Probability for a specific node
        for cpd in cpds:
            if cpd.variable == node_key:
                cpd_table = cpd_to_df(cpd)
                st.dataframe(cpd_table)

        pred = get_sim_y(node_key, window)
        Posterior(pred)
        Forest(pred)
        Summary(pred)

def main():
    run()

if __name__ == '__main__':
    main()