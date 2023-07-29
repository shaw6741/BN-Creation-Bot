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
from pgmpy.inference import VariableElimination

engine = Engine()
engine.start()
model = engine.BN_model.model
value_meanings_dic = engine.BN_model.value_meanings
for node in engine.BN_model.model.nodes:
    if node in value_meanings_dic:
        model.nodes[node]['state_names'] = value_meanings_dic[node]

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
st.sidebar.divider()
node_key = find_key(new_dict, node_value)
child_node = get_child_node(node_key)
node_value_meanings = value_meanings_dic[node_key]

st.sidebar.markdown('**States for this node**')
states1 = model.nodes[node_key]['state_names']
for key, value in states1.items():
    st.sidebar.markdown(f"**_{value.split(':')[0]}_ means _{value.split(':')[1]}_**")

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
    cpds = engine.BN_model.get_cpds()

if any(conditions): # have historical data
    cpd = engine.BN_model.model.get_cpds(node_key)
    print(model.nodes[node_key])
    cpd_table, parent_node_states = cpd_to_df(cpd, node_key, value_meanings_dic)
    # print(parent_node_states)
    for col in cpd_table.columns:
        if col in value_meanings_dic:
            cpd_table[col] = cpd_table[col].astype(str).replace(value_meanings_dic[col])
    
    if cpd_table.shape[1] == 2:
        get_cpd_agg(cpd_table)
    
    if cpd_table.shape[1] > 2:
        cpd_form, evidence_form = st.columns([5,3]) 
        with cpd_form:
            get_cpd_agg(cpd_table)
            
        with evidence_form:
            st.markdown('**Enter new evidence to see how the predicted probability change**')
            
            select_evidence = st.multiselect('Evidence', parent_node_states, key='evidence')
        
            evidence = {state.split(':')[0].split('_')[0]: int(state.split(':')[0].split('_')[1]) for state in st.session_state['evidence']}

            infer = VariableElimination(engine.BN_model.model)
            infer_result = infer.query([node_key], evidence=evidence)
            
            infer_df = {f'{node_key}': [], f'Prob({node_key})': []}
            state_names = infer_result.state_names[node_key]
            for state, prob in zip(state_names, infer_result.values):
                infer_df[f'{node_key}'].append(f"{state}")
                infer_df[f'Prob({node_key})'].append(prob)
            infer_df = pd.DataFrame(infer_df)

            infer_gb = GridOptionsBuilder.from_dataframe(infer_df)
            infer_gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, editable=False)
            infer_gb.configure_column(f'Prob({node_key})', type=["numericColumn","numberColumnFilter","customNumericFormat"], precision=3, aggFunc='sum')
            infer_gb.configure_grid_options(domLayout='normal')
            infer_gridOptions = infer_gb.build()
            infer_df_agg = AgGrid(
                    infer_df, 
                    gridOptions=infer_gridOptions,
                    height=200,  width='100%',
                    columns_auto_size_mode='FIT_ALL_COLUMNS_TO_VIEW',
                    editable=False
                    )

else: # no historical data
    st.markdown('**We don\'t have historical data for this node.\
                 Change the state & probabilities to see how the result changes.**')

    file_path = f"./engine/{node_key}_probs.csv"
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
