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

def get_prior_probs_template(missing_nodes_name):
    template = f"""
        I want you to act as a person collecting information on the probability and relative weight.
        Ask questions one at a time, sequentially!
        Wait for my answer before moving on to the next question.
        Always start the conversation with 'Hi, let's get the prior probabilities for some missing nodes. Shall we start?'
        Below is the instruction on asking questions.
            
        Q1: "Which state has a lower probability: "{missing_nodes_name}" or Not {missing_nodes_name}?"

        User provide answers.

        Bot: Thank you for your response. Now, moving on to the next question. Based on the lower probability state,

        Q2: "If relative weight of (the lower probability state from Q1) is 1, what is the relative weight of (opposite of lower probability state from Q1)?"

        User provide answers.

        Bot: Thank you for your answers. Now, let's summarize the information on relative weight values.
        {missing_nodes_name} = Relative weight numeric value
        Not {missing_nodes_name} = Relative weight numeric value
        Thank you. I will start next step.

        Example summary:
        {missing_nodes_name} = 1
        Not {missing_nodes_name} = 3
        Thank you. I will start the next step.

        Always end your summary with: 'Thank you. I will start the next step.'

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


def calculate_max_eigenvector(numeric_values):
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


#  """0: not mc, 1: mc of 5%, 2 mc of 10% and 3 mc of 20%"""
def get_cond_probs_template(missing_nodes_name):
    template = f"""
        You will act as a person collecting information on the probability and relative weight. 
        You will ask Q1 ~ Q8 questions one at a time, sequentially. 
        Please wait for user's answer before moving on to the next question.
        Always start the conversation with 'Thank you for finishing the prior probabilities part, next let\'s get the conditional probabilities for some missing nodes with its child node. Shall we start?'
        
        Instruction on asking questions:
        Below are eight different questions. Before you answer, remember that the value of the probability should be between 0 and 1.

        Q1: "What are your opinions on the lower limit, upper limit, and most likely value(in percentage) of 'Investment loss', considering the occurrence of {missing_nodes_name} and no market correction happening?"

        User: [Provide your answer to Q1]

        Bot: Thank you for your response. Now, moving on to the next question.

        Q2: "What are your opinions on the lower limit, upper limit, and most likely value(in percentage) of 'Investment loss', considering the occurrence of {missing_nodes_name} and market correction of 5% happening?"

        User: [Provide your answer to Q2]

        Bot: Thank you for your response. Now, moving on to the next question.

        Q3: "What are your opinions on the lower limit, upper limit, and most likely value(in percentage) of 'Investment loss', considering the occurrence of {missing_nodes_name} and market correction of 10% happening?"

        User: [Provide your answer to Q3]

        Bot: Thank you for your response. Now, moving on to the next question.

        Q4: "What are your opinions on the lower limit, upper limit, and most likely value(in percentage) of 'Investment loss', considering the occurrence of {missing_nodes_name} and market correction of 20% happening?"

        User: [Provide your answer to Q4]

        Bot: Thank you for your response. Now, moving on to the next question.

        Q5: "What are your opinions on the lower limit, upper limit, and most likely value(in percentage) of 'Investment loss', considering the occurrence of not {missing_nodes_name} and no market correction happening?"

        User: [Provide your answer to Q5]

        Bot: Thank you for your response. Now, moving on to the next question.

        Q6: "What are your opinions on the lower limit, upper limit, and most likely value(in percentage) of 'Investment loss', considering the occurrence of not {missing_nodes_name} and market correction of 5% happening?"

        User: [Provide your answer to Q6]

        Bot: Thank you for your response. Now, moving on to the next question.

        Q7: "What are your opinions on the lower limit, upper limit, and most likely value(in percentage) of 'Investment loss', considering the occurrence of not {missing_nodes_name} and market correction of 10% happening?"

        User: [Provide your answer to Q7]

        Bot: Thank you for your response. Now, moving on to the next question.

        Q8: "What are your opinions on the lower limit, upper limit, and most likely value(in percentage) of 'Investment loss', considering the occurrence of not {missing_nodes_name} and market correction of 20% happening?"

        User: [Provide your answer to Q8]

        Bot: Thank you for your answers. Now, let's summarize the information on relative weight values.

        Summary:
        Q1 = [probability lower limit], [probability upper limit], [most likely value from Q1]
        Q2 = [probability lower limit], [probability upper limit], [most likely value from Q2]
        Q3 = [probability lower limit], [probability upper limit], [most likely value from Q1]
        Q4 = [probability lower limit], [probability upper limit], [most likely value from Q2]
        Q5 = [probability lower limit], [probability upper limit], [most likely value from Q1]
        Q6 = [probability lower limit], [probability upper limit], [most likely value from Q2]
        Q7 = [probability lower limit], [probability upper limit], [most likely value from Q2]
        Q8 = [probability lower limit], [probability upper limit], [most likely value from Q2]

        For example:
        Q1 = 10, 30, 15
        Q2 = 30, 10, 20
        Q3 = 5, 10, 20
        Q4 = 7, 10, 25
        Q5 = 0, 10, 20
        Q6 = 20, 10, 80
        Q7 = 10, 10, 50
        Q8 = 40, 10, 70

        Always end your summary with: 'Thank you. I'\ll start creating'

        Remember to ask the question one at a time, sequentially. 
        """
    return template

def extract_opinions_list(assistant_response):
    pattern = r'Q\d\s=\s([\d.,\s]+)'
    match = re.findall(pattern, assistant_response)
    opinions_list = [list(map(float, values.split(', '))) for values in match]
    return opinions_list

def triangular_cdf(x, a, b, c):
    if x <= a:
        return 0.0
    elif a < x <= c:
        return (x - a)**2 / ((b - a) * (c - a))
    elif c < x < b:
        return 1 - (b - x)**2 / ((b - a) * (b - c))
    else:
        return 1

def calculate_probability_at_x(opinions_list, x):
    probabilities = []
    for opinion_list in opinions_list:
        a, b, c = opinion_list
        probability_at_x = triangular_cdf(x, a, b, c)
        probabilities.append(probability_at_x)
    return probabilities

def create_result_array(assistant_response, x_value):
#def create_result_array(opinions_list, x_value):
    opinions_list = extract_opinions_list(assistant_response)
    
    # Calculate the probabilities at the X value for each distribution
    probability_at_x = calculate_probability_at_x(opinions_list, x_value)
    probability_at_x_compl = [1 - prob for prob in probability_at_x]
    
    result_array = np.array([probability_at_x, probability_at_x_compl])
    return result_array