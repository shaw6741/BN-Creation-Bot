import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import json
from engine.Engine import Engine
from utils.utils import call_arviz_lib

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
    st.header('Distribution')
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

def find_key(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None

def cpd_to_df(cpd):
    variables = cpd.variables
    cardinality = cpd.cardinality
    values = cpd.values.flatten()

    # Generate all combinations of variable states
    index_tuples = pd.MultiIndex.from_product([range(card) for card in cardinality], names=variables)

    # Create DataFrame
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

def create_prob_table():
    df = pd.DataFrame(
        [
            {"State": None, "Prob": None},
            {"State": None, "Prob": None},
        ]
    )
    st.data_editor(df, key="data_editor") # 👈 Set a key
    st.write("Here's the session state:")
    st.write(st.session_state["data_editor"]) # 👈 Access the edited data
    
def run():
    # Setting page title and header
    st.set_page_config(page_title="BN Results", page_icon=":robot_face:")  # , layout="wide"
    st.title("Bayesian Network Results")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    path = './engine/conversation.json'
    with open(path, 'r') as file:
            json_file = json.load(file)
    
    new_dict = {}
    for section in json_file:
        if isinstance(json_file[section], dict):
            md_text = "- **{}**: ".format(section.title())
            md_text += "; ".join(["{} ({})".format(value, key) for key, value in json_file[section].items()])
            st.markdown(md_text)
            new_dict.update(json_file[section])

    # Sidebar to create
    st.sidebar.title('Select Variable')
    window = st.sidebar.slider('Num Samples', min_value=1000, max_value=100000, step=500)
    node_value = st.sidebar.selectbox('Node', list(new_dict.values()))
    node_key = find_key(new_dict, node_value)

    intro = st.container()
    with intro:
        Network()

    body = st.container()
    with body:
        cpds = engine.BN_model.get_cpds()     
        st.markdown("""### Node: {}""".format(node_value))
        create_prob_table()

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