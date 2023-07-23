import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from engine.Engine import Engine
from utils.utils import call_arviz_lib
from utils.chat_help import *
from utils.visual_help import *
import pandas as pd

engine = Engine()
engine.start()

def get_sim_y(node, n_samples):
    predict_df = engine.BN_model.model_simulate(n_samples)
    predict_ = predict_df[node]
    return predict_df, predict_.values

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

def Autocorrelation(predict_y):
    st.markdown('#### Autocorrelation')
    st.pyplot(call_arviz_lib().get_autocorrelation(predict_y))

def Trace(predict_y):
    st.markdown('#### Trace')
    st.pyplot(call_arviz_lib().get_plot_trace(predict_y))

def MCSE(predict_y):
    st.markdown('#### MCSE')
    st.pyplot(call_arviz_lib().get_plot_mcse(predict_y))

def Log_likelihood(df):
    st.markdown('#### log likelihood')
    st.write(engine.BN_model.get_log_l_score(df))

def K2(df):
    st.markdown('#### K2 Score')
    st.write(engine.BN_model.get_k2_est(df))

#def run():
# Setting page title and header
st.set_page_config(page_title="BN Results", page_icon=":robot_face:", layout='wide')  # , layout="wide"
st.title("Bayesian Network Results")
st.set_option('deprecation.showPyplotGlobalUse', False)

new_dict = get_node_fullname()

# Sidebar
st.sidebar.title('Select Variable')
window = st.sidebar.slider('Num Samples', min_value=1000, max_value=100000, step=500)
node_value = st.sidebar.selectbox('Node', list(new_dict.values()))
node_key = find_key(new_dict, node_value)
child_node = get_child_node(node_key)

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
                        for keyword in ['market correc', 'investment los',
                                        'portfolio loss',
                                        ]),
                ]


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
    st.markdown('**Prior & Conditional Probabilities**')

    file_path = f"./engine/{node_key}_probs.csv"  # Replace with the path to your file

    if not os.path.exists(file_path):
        get_prob_table0(node_key, child_node)
    else:
        get_prob_table1(node_key)
        
# ---------------------
# FINAL PLOTS
st.divider()
visuals = st.container()
with visuals: # for a specific node
    df_pred, pred = get_sim_y(node_key, window)
    Posterior(pred)
    Forest(pred)
    Summary(pred)
    Autocorrelation(pred)
    Trace(pred)
    MCSE(pred)
    Log_likelihood(df_pred)
    K2(df_pred)
