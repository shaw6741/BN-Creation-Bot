import re, json, textwrap, openai
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import messages_from_dict, messages_to_dict
from langchain.callbacks import get_openai_callback

import streamlit as st

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
             component_dic = {item.split(" ")[-1].replace("(", "").replace(")", ""): item.split(" ")[0] for item in component.split(", ")}
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



text = "Ok, so here's a summary of the information you provided: \n1. Triggers: inflation (IN), trade war (TW), declaration of war (DW)\n2. Controls: None\n3. Risk Events: market correction (MC)\n4. Mitigators: buying straddles (BS)\n5. Consequences: return loss (RL)\n6. Edges: IN -> MC, TW -> MC, DW -> MC, BS -> RL, MC -> RL"

# Extract Triggers
triggers_start_index = text.index("Triggers: ") + len("Triggers: ")
triggers_end_index = text.index("\n", triggers_start_index)
triggers_text = text[triggers_start_index:triggers_end_index].split(", ")
triggers = {trigger.split(" ")[-1].replace("(", "").replace(")", ""): trigger.split(" ")[0] for trigger in triggers_text}