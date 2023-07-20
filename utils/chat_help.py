import re, json, textwrap, openai, os
import numpy as np
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import messages_from_dict, messages_to_dict
from langchain.callbacks import get_openai_callback
import pandas as pd
import streamlit as st

# Template for main chat
def get_template():
    template = """
    You're a robot collecting info from the user about making a Bayesian Network to assess the risk of event(s).
    You need to gather info about 5 things - risk event(s), trigger(s), control(s), consequence(s), mitigator(s).
    Ask questions one by one to collect the required info from the user.
    You should always start by asking about the risk event(s).
    After getting all the required info, summarize it as follows:
    1. Triggers: ... (abbr), ...
    2. Controls: ... (abbr), ...
    3. Risk Events: ... (abbr), ...
    4. Mitigators: ... (abbr), ...
    5. Consequences: ... (abbr), ...
    6. Edges: ... -> ..., ... -> ...
    Always end the summary with 'Thank you. I'll start creating...'. Always separate by comma.
    Put 'None' if there's no info about a certain component.
    Remember that triggers and controls are parent nodes for risk events, risk events and controls are parent nodes for consequences!
    And don't confuse controls and mitigators! Controls are for risk events, while mitigators are for consequences!

    An example summary is:
    1. Triggers: some outbreaks (SO), supply chain (SC)
    2. Controls: abc (ABC), cde (CDE)
    3. Risk Events: cbd (CBD), DFF (DF)
    4. Mitigators: AB (AB), BCD (BCD), EEEF (EF)
    5. Consequences: BCCC (BC), asdkf (ASD)
    6. Edges: SO -> CBD, SC -> CBD, SC -> DF, AB -> BC, BCD -> BC, EF -> ASD, CBD -> BC, CBD -> ASD, DF -> ASD
    Thank you. I'll start creating...

    Current conversation:
    {history}

    Human: {input}
    Assistant:
    """

    PROMPT = PromptTemplate(
            input_variables=["history", "input"],
            template=template
                )
    
    return template, PROMPT

# ---------------------------------------------
# Format Results
def get_node_names(texts, component_keyword):
    """
    Extracts node names from the given texts based on the component keyword.

    texts (str): The summarized text from chat.
    component_keyword (str): The component, 'Triggers', 'Controls', 'Risk Events', 'Mitigators', 'Consequences'

    Returns:
        component_dic (dict): A dictionary of node names with their corresponding identifiers, or None if no nodes are found.

    """
    pattern = r"{}: (.+?)\n".format(component_keyword)
    matches = re.search(pattern, texts)
    component_dic = None
    if matches:
        component = matches.group(1)
        if component != 'None':
             component_dic = {item.split(" ")[-1].replace("(", "").replace(")", ""): item.split(" (")[0] for item in component.split(", ")}
    return component_dic

def get_edges(texts):
    edges = []
    edges_pattern = r"Edges:\s*(.*)"
    edges_match = re.search(edges_pattern, texts)
    if edges_match:
        edges_text = edges_match.group(1)
        edges_list = re.findall(r"(\w+) -> (\w+)", edges_text)
        edges = [(parent, child) for parent, child in edges_list]
    return edges

def format_data(response):
        triggers = get_node_names(response, 'Triggers')
        controls = get_node_names(response, 'Controls')
        events = get_node_names(response, 'Risk Events')
        mitigators = get_node_names(response, 'Mitigators')
        consequences = get_node_names(response, 'Consequences')
        edges = get_edges(response)

        data = {
            'triggers': triggers, 'controls': controls,
            'events': events, 'mitigators': mitigators,
            'consequences': consequences, 'edges': edges,
            'mkt_cap': st.session_state['mkt_cap'][-1],
            'style': st.session_state['style'][-1],
            'sectors': st.session_state['sectors'][-1],
            'hedge': st.session_state['hedge'][-1],
            'long/short': st.session_state['longshort'][-1],
        }

        return data


# ------------------------------
# To find tickers for the nodes, based on economic indicator dic we have
template_find_tickers = """
Find the ticker values for {} based on the dictionary provided {}.
Return me a python dictionary,
keys are the variables, values are the tickers.
If no ticker found, then put variable:None.
Start the answer with the python dictionary. DO NOT add any more words beside the dictionary.
"""

from utils.utils import get_data
def chat_find_tickers(variables):
    ticker_dic = get_data().economical_ticker_dict
    ticker_dic.update({'declaration of war':'DW','market correction':'MC'})
    user_message = template_find_tickers.format(variables, ticker_dic)
    
    messages = []
    messages.append({"role": "user", "content": user_message})
    completion = openai.ChatCompletion.create(
            #model="gpt-3.5-turbo-0613",
            model = 'gpt-3.5-turbo-16k',
            messages=messages,
            temperature=0,
        )

    response = completion.choices[0].message.content
    return response