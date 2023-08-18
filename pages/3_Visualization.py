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
import arviz as az

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

def convert_to_inf_data(node_key, predict_y):
    data_dict = az.dict_to_dataset({node_key:predict_y})
    inf = az.convert_to_dataset(data_dict)
    return inf

def Network():
    st.markdown('#### Network Plot')
    fig = engine.BN_model.plot_networkx()
    st.write(fig)

def Distribution(predict_y):
    st.markdown('#### Distribution')
    obj = call_arviz_lib().get_dist(predict_y)
    st.pyplot(obj)

def Posterior(inf_y, node_key):
    st.markdown('#### Posterior Distribution')
    st.pyplot(call_arviz_lib().get_posterior(inf_y, node_key))

def Forest(node_key, inf_y):
    explain = """
    Point estimate, usually the mean/median of the posterior distribution.\n\n
    ESS(effective sample size): amount of info contained in the posterior distribution, 
    represents the sample size that would provide the same aount of info 
    if the posterior distribution were a simple random sample.
    Larger ESS -> more reliable and precise estimates.\n\n
    R hat: assess the convergence of multiple MCMC chain. 
    No R hat = a single MCMC chain was used or 
    there's no need for multiple chains because the model has already converged.

    """
    st.markdown('#### Plot Forest', help = explain)
    st.pyplot(call_arviz_lib().get_plot_forest(node_key, inf_y))

def Summary(predict_y):
    st.markdown('#### Summary Table')
    df = pd.DataFrame(call_arviz_lib().get_summary(predict_y))  # st.dataframe
    st.write(df)

def Autocorrelation(predict_y):
    acf_explain = """
    1. Convergence: in ACF, convergence is reflected in how quickly the autocorrelation decreases as the lag increases. 
    If the autocorrelation decays rapidly and approaches zero, it indicates that the MCMC chain has converged to the target distribution. 
    Smaller autocorrelation values suggest better convergence.\n\n
    2. Mixing: Mixing relates to the decorrelation of samples at different lags.
    If the ACF plot shows slow decay and lags with relatively high autocorrelation, it indicates poor mixing.
    It means the chain is not exploring the posterior distribution effectively and is getting stuck in certain regions.\n\n
    3. Burn-in: The ACF plot alone may not directly indicate the burn-in period.
     However, in conjunction with the Trace plot, you can identify a suitable burn-in phase where the chain stabilizes and reaches the target distribution.
    The burn-in period consists of the initial samples that are discarded before convergence.
    """
    st.markdown('##### Autocorrelation', help = acf_explain)
    st.pyplot(call_arviz_lib().get_autocorrelation(predict_y))

def Trace(predict_y):
    trace_explain = """
    1. Convergence: convergence is observed as a stable meandering pattern around regions of high posterior density. 
    When the chain has converged, the sampled values should appear stationary, without any significant trends or drifts.\n\n
    2. Mixing: Mixing is evident when the Trace plot shows good exploration of the posterior space. 
    Samples should appear to be drawn from different areas of the distribution in a random and well-distributed manner, indicating good mixing.\n\n
    3. Burn-in: By observing the Trace plot, you can identify the iterations before which 
    the chain appears to be exploring an area far from the target distribution. 
    These initial iterations represent the burn-in period and should be discarded.\n\n
    4. Sample variability: The spread of the sampled values on the Trace plot represents the sample variability. 
    A wider spread indicates higher uncertainty in the parameter estimates, 
    while a narrower spread suggests more precise estimates.\n\n
        
    """
    st.markdown('##### Trace', help=trace_explain)
    st.pyplot(call_arviz_lib().get_plot_trace(predict_y))

def MCSE(predict_y):
    mcse_explain = """
    MCSE = Monte Carlo Standard Error
    - MCSE is like a measure of the uncertainty or "error bars" around our estimates.
    - The MCSE plot shows how much our estimates could vary due to limited data.
    - A smaller MCSE value means our estimates are more precise and reliable.
    - A larger MCSE value indicates more uncertainty and less precise estimates.
    - The MCSE plot helps us understand the quality of our results and whether we need more data to improve accuracy.
    """
    st.markdown('#### MCSE', help = mcse_explain)
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

json_file, new_dict, md_text_lst = get_node_fullname()

node_val_lst = ['all']
for val in list(new_dict.values()):
    node_val_lst.append(val)

# Sidebar
st.sidebar.title('Select Variable')
node_value = st.sidebar.selectbox('Node', node_val_lst)
window = st.sidebar.slider('Num Samples', min_value=1000, max_value=100000, step=500)

if node_value == 'all':
    write_conversation_info(new_dict, md_text_lst)
    df_pred, pred = get_sim_y(find_key(new_dict, node_val_lst[-1]), n_samples=window)
    ll = engine.BN_model.get_log_l_score(df_pred)
    k2 = engine.BN_model.get_k2_est(df_pred)
    ll_explain = """
    Measure of how well the entire Bayesian network model fits the observed data.
    It quantifies the likelihood of seeing the observed data given the model's structure and parameter estimates.
    The higher the better.
    """
    k2_explain = """
    Assess the fitness of the Bayesian network model's conditional probability distributions.
    It evaluates how well the model's conditional probabilities align with the observed data and the dependencies among the variables in the network.
    The higher the better.
    """
    st.markdown(f'- **Log-likelihood**: {ll}', help=ll_explain)
    st.markdown(f'- **K2 Score**: {k2}', help = k2_explain)
    Network()    
    

else:
    st.sidebar.divider()
    
    node_key = find_key(new_dict, node_value)
    child_node = get_child_node(node_key)
    node_value_meanings = value_meanings_dic[node_key]

    st.sidebar.markdown('**States for this node**')
    states1 = model.nodes[node_key]['state_names']
    for key, value in states1.items():
        st.sidebar.markdown(f"**_{value.split(':')[0]}_ means _{value.split(':')[1]}_**")
    

    # -------------------
    # CHECK PROBABILITIES
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
        inf = convert_to_inf_data(node_key, pred)
        Posterior(inf, node_key)
        Summary(pred)
        Forest(node_key, inf)
        st.markdown('#### ACF & Trace')        
        act, trace = st.columns(2)
        with act:
            Autocorrelation(pred)
        with trace:
            Trace(pred)

        MCSE(pred)
        
