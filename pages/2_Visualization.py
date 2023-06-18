import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
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

def run():
    # Setting page title and header
    st.set_page_config(page_title="BN Results", page_icon=":robot_face:")  # , layout="wide"
    st.title("Bayesian Network Results")
    # st.subheader('Powered by OpenAI + Streamlit')
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Sidebar to create
    st.sidebar.title('User Variable')
    window = st.sidebar.slider('Number Samples', min_value=1000, max_value=100000, step=500)
    node = st.sidebar.selectbox('Nodes', list(engine.BN_model.get_nodes()))

    intro = st.container()
    with intro:
        Network()

    body = st.container()
    with body:
        pred = get_sim_y(node, window)

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