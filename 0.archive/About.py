# home page for streamlit
import streamlit as st

st.set_page_config(page_title="BN Bot", page_icon=":robot_face:")
st.title("Bayesian Network Creation App")

st.markdown(
        """
        **Bayesian Network** is widely used in risk assessment. 
        It is a probabilistic model that describes the probability of an event occurring in a given situation.

        With our app, you don't need to know the details of the Bayesian Network to create it.
        We instruct *ChatGPT* to guide you through the creation process.

        **Let's go to the Chat page and get started.**

        If you'd like to learn more:
        - Check out what is a [Bayesian Network](https://en.wikipedia.org/wiki/Bayesian_network)
        - Powered by OpenAI's [ChatGPT](https://www.openai.com/blog/chatgpt/)
            - GPT Model Used: [gpt-3.5-turbo](https://platform.openai.com/docs/models/gpt-3-5)
        - Data Source: [Yahoo Finance](https://finance.yahoo.com/) & [FRED](https://fred.stlouisfed.org/).

        """
        
)
