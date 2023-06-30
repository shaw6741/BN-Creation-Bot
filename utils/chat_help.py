import re, json, textwrap, openai
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import messages_from_dict, messages_to_dict
from langchain.callbacks import get_openai_callback

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
Always ask consequence before mitigator!

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

# Format Results
def get_node_names(texts, component_keyword):
    """
    Extracts node names from the given texts based on the component keyword.

    texts (str): The summarized text from chat.
    component_keyword (str): The component, 'Triggers', 'Controls', 'Risk Events', 'Mitigators', 'Consequences'

    Returns:
        component_dic (dict): A dictionary of node names with their corresponding identifiers, or None if no nodes are found.

    """
    pattern = r"{}: (.+)".format(component_keyword)
    matches = re.search(pattern, texts)
    component_dic = None
    if matches:
        nodes = matches.group(1)
    if nodes != 'None':
        component_dic = {item.split(' (')[1][:-1].strip(): item.split(' (')[0].strip() for item in nodes.split(', ')}

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
