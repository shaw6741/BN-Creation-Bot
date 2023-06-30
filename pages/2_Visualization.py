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
    st.header('Network Plot')
    fig = engine.BN_model.plot_networkx()
    st.write(fig)

def Distribution(predict_y):
    st.header('Distribution')
    obj = call_arviz_lib().get_dist(predict_y)
    st.pyplot(obj)

def Posterior(predict_y):
    st.header('Posterior Distribution')
    st.pyplot(call_arviz_lib().get_posterior(predict_y))

def Forest(predict_y):
    st.header('Plot Forest')
    st.pyplot(call_arviz_lib().get_plot_forest(predict_y))

def Summary(predict_y):
    st.header('Summary Table')
    df = pd.DataFrame(call_arviz_lib().get_summary(predict_y))  # st.dataframe
    st.write(df)

def find_key(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None


def run():
    # Setting page title and header
    st.set_page_config(page_title="BN Results", page_icon=":robot_face:")  # , layout="wide"
    st.title("Bayesian Network Results")
    st.set_option('deprecation.showPyplotGlobalUse', False)
    path = 'E:/px/UChi/Courses/Capstone/BN-Creation-Bot/engine/conversation.json'
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
    st.sidebar.title('User Variable')
    window = st.sidebar.slider('Num Samples', min_value=1000, max_value=100000, step=500)
    node_value = st.sidebar.selectbox('Node', list(new_dict.values()))
    node_key = find_key(new_dict, node_value)

    intro = st.container()
    with intro:
        Network()

    body = st.container()
    with body:
        pred = get_sim_y(node_key, window)

        #col1, col2, col3 = st.columns(3)
        # Distribution(pred)
        #with col1:
        Posterior(pred)
        #with col2:
        Forest(pred)
        #with col3:
        Summary(pred)

def main():
    run()

if __name__ == '__main__':
    main()