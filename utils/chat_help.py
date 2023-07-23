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

def get_probs_template():
    template = """
        I want you to act as a person collecting information on the probability and relative weight.
        Ask questions one at a time, sequentially!
        Wait for my answer before moving on to the next question.

        Below is the instruction on asking questions.
        Q1. Probability of which state is lower: "low" or "medium" for {missing_nodes_name}?
        Q2. The follow-up question you should ask is;
        Based on the lower probability state, you should ask 
        "If relative weight of (previous answer) is 1, what is the relative weight of (opposite of previous answer)?". 

        Q3. Probability of which state is lower: "medium" or "high" for {missing_nodes_name}?
        Q4. The follow-up question you should ask is;
        Based on the lower probability state, you should ask 
        "If relative weight of (previous answer) is 1, what is the relative weight of (opposite of previous answer)?". 

        Q5. Probability of which state is lower: "low" or "high" for {missing_nodes_name}?
        Q6. The follow-up question you should ask is;
        Based on the lower probability state, you should ask 
        "If relative weight of (previous answer) is 1, what is the relative weight of (opposite of previous answer)?". 


        After you've collected all the information on relative weight value, summarize as follows:
        low_mid = Relative weight numeric value from Q2
        mid_high = Relative weight numeric value from Q4
        low_high = Relative weight numeric value from Q6

        An example summary is as follows;
        low_mid = 1.5
        mid_high = 4
        low_high = 7

        Always end the summary with: 'Thank you. I\'ll start creating...'

        Current conversation:
        {history}

        Human: {input}
        Assistant:

        """

    PROMPT = PromptTemplate(
            input_variables=['missing_node_name', "history", "input"],
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
Find the ticker values for this node dictionary {} based on this ticker dictionary {}.
The node name may be different with the ticker name, but you should find it based on your understanding of the node and the ticker.
Return me a python dictionary,
keys are the variables, values are the tickers.
If no ticker found, then put variable:None.
Start the answer with the python dictionary. DO NOT add any more words beside the dictionary.
"""

from utils.utils import get_data
def chat_find_tickers(variables):
    # get API key
    api_path = "./pages/API_O.txt"
    with open(api_path, "r") as file:
        API_O = file.read()
        openai.api_key = API_O
    ticker_dic = get_data().economical_ticker_dict
    ticker_dic.update(get_data().market_ticker_dict)
    ticker_dic.update({'declaration of war':'DW','market correction':'MC'})
    ticker_dic.update({'trade war':'DW','investment loss':'IL', 'portfolio loss':'PL'})
    ticker_dic.update({'oil supply':'CL=F'})
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

def get_probs_template(missing_nodes_name):
    template = f"""
            I want you to act as a person collecting information on the probability and relative weight.
            Ask questions one at a time, sequentially!
            Wait for my answer before moving on to the next question.

           Below is the instruction on asking questions.
            
        Q1: "CWhich state has a lower probability: "{missing_nodes_name}" or Not {missing_nodes_name}?"

        User: [Provide your answer to Q1]

        Bot: Thank you for your response. Now, moving on to the next question. Based on the lower probability state,

        Q2: "If relative weight of (previous answer from Q1) is 1, what is the relative weight of (opposite of previous answer from Q1)?"

        User: [Provide your answer to Q2]

        Bot: Thank you for your answers. Now, let's summarize the information on relative weight values.

            After you've collected all the information on relative weight value, summarize as follows:
            {missing_nodes_name} = Relative weight numeric value
            Not {missing_nodes_name} = Relative weight numeric value

            An example summary is as follows;
            {missing_nodes_name} = 1
            Not {missing_nodes_name} = 3

            Always end your summary with: 'Thank you. I'\ll start creating'

            """
    # conversation_history = """
    #     Current conversation:
    #     {history}

    #     Human:{input}
    #     Assistant
    #     """

    # template = template + conversation_history

    # PROMPT = PromptTemplate(
    #         input_variables=["history", "input"],
    #         template=template
    #             )
    
    return template


def calculate_max_eigenvector(assistant_response):
    numeric_values = [float(value) for value in re.findall(r"=\s(\d+\.\d+|\d+)", assistant_response)]

    if numeric_values[0] == 1:
        nested_array = [
            [1, 1/numeric_values[1]],
            [numeric_values[1], 1],
        ]
    else:
        nested_array = [
            [1, numeric_values[0]],
            [1/numeric_values[0], 1],
        ]

    matrix = np.array(nested_array)
    
    eigenvalues, eigenvectors = np.linalg.eig(matrix)

    max_eigenvalue_index = np.argmax(eigenvalues)
    max_eigenvalue = eigenvalues[max_eigenvalue_index]

    max_eigenvector = np.real(eigenvectors[:, max_eigenvalue_index])

    # Normalize the eigenvector
    max_eigenvector /= np.sum(max_eigenvector)
    max_eigenvector_2d = np.array([[value] for value in max_eigenvector])


    return matrix, max_eigenvector_2d


def get_cond_probs_template(missing_nodes_name):
    template = f"""
        You will act as a person collecting information on the probability and relative weight. 
        You will ask Q1 and Q2 questions one at a time, sequentially. 
        Please wait for user's answer before moving on to the next question.

        Instruction on asking questions:
        Below are two different questions. Before you answer, remember that the value of the probability should be between 0 and 1.

        Q1: "Considering {missing_nodes_name} occurs and a 15% 'Investment loss' over 5 days, what do you think are the most likely lower limit, upper limit, and mode of the probability?"

        User: [Provide your answer to Q1]

        Bot: Thank you for your response. Now, moving on to the next question.

        Q2: "Considering {missing_nodes_name} does not occur and a 15% 'Investment loss' over 5 days, what do you think are the most likely lower limit, upper limit, and mode of the probability?"

        User: [Provide your answer to Q2]

        Bot: Thank you for your answers. Now, let's summarize the information on relative weight values.

        Summary:
        Q1 = [probability lower limit], [probability upper limit], [most likely value from Q1]
        Q2 = [probability lower limit], [probability upper limit], [most likely value from Q2]

        For example:
        Q1 = 0.1, 0.3, 0.4
        Q2 = 0.2, 0.4, 0.3
        
        Always end your summary with: 'Thank you. I'\ll start creating'

        """
    return template