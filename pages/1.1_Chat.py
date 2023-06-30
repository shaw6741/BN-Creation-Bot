import openai
import streamlit as st
from streamlit_chat import message
import json, pickle, re

# Setting page title and header
st.set_page_config(page_title="BN Chatbot", page_icon=":robot_face:")
st.title("Bayesian Network Creation Chat")
st.markdown("""
**Instruction**:
- Enter your OpenAI API Key in the sidebar.
- Select your portfolio information.
- Say *Hi* to ChatGPT and let's get started!
""")

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
"""

# Initialise session state variables
if 'generated' not in st.session_state:
    st.session_state['generated'] = []
if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'messages' not in st.session_state:
    st.session_state['messages'] = []
    #st.session_state['messages'].append({'role':'system', 'content':template})

if 'total_tokens' not in st.session_state:
    st.session_state['total_tokens'] = []

if 'mkt_cap' not in st.session_state:
    st.session_state['mkt_cap'] = []
if 'style' not in st.session_state:
    st.session_state['style'] = []
if 'sectors' not in st.session_state:
    st.session_state['sectors'] = []
if 'hedge' not in st.session_state:
    st.session_state['hedge'] = []
if 'longshort' not in st.session_state:
    st.session_state['longshort'] = []
if 'conversation_ended' not in st.session_state:
    st.session_state['conversation_ended'] = False

if 'stored_session' not in st.session_state:
    st.session_state["stored_session"] = []

def get_text():
    """
    Get the user input text.

    Returns:
        (str): The text entered by the user
    """
    input_text = st.text_input("You: ", st.session_state["messages"], key="input",
                               placeholder="Your AI assistant here! Talk with me ...", 
                               label_visibility='hidden')
    return input_text

# Define function to start a new chat
def new_chat():
    """
    Clears session state and starts a new chat.
    """
    save = []
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        save.append("User:" + st.session_state["past"][i])
        save.append("Bot:" + st.session_state["generated"][i])        
    st.session_state["stored_session"].append(save)
    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["messages"] = []
    st.session_state['total_tokens'] = []
    st.session_state['mkt_cap'] = []
    st.session_state['style'] = []
    st.session_state['sectors'] = []
    st.session_state['hedge'] = []
    st.session_state['longshort'] = []
    st.session_state['conversation_ended'] = False

# Sidebar - API key, portfolio market cap, style, sectors
st.sidebar.title("Info Needed")

# OpenAI API key
API_O = st.sidebar.text_input('OPENAI-API-KEY', type='password')
if API_O:
    openai.api_key = API_O
else:
    st.sidebar.warning(
        'API key required.The API key is not stored.'
    )

# portfolio related
sectors = st.sidebar.multiselect('Sectors',
                               ['SP','NASDAQ','DOW','Russell',
                                'CRUDE', 'OIL', 'GOLD', 'SILVER',
                                'BOND', 'NOTE', 'FED',
                                'Financial', 'Materials', 'Communications', 'Energy', 'Industrial',
                                'Technology', 'Consumer', 'Real_Estate', 'Utilities', 'Healthcare', 'Consumer',
                               ])

mkt_cap = st.sidebar.selectbox('Market Cap',
                               ('Large_Cap','Mid_Cap','Small_Cap'))

style = st.sidebar.selectbox('Style',
                             ('Growth','Value','Core'))

hedge = st.sidebar.selectbox('Hedge?',
                             ('Hedged','Not Hedged'))

longshort = st.sidebar.selectbox('Long or Short?',
                             ('Long','Short'))

st.sidebar.button("New Chat", on_click = new_chat, type='primary')

# generate a response
def generate_response(prompt):
    st.session_state['messages'].append({"role": "user", "content": prompt})
    st.session_state['messages'].append({'role':'system', 'content':template})
    
    completion = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=st.session_state['messages'],
        temperature=0.3,
    )
    
    response = completion.choices[0].message.content
    st.session_state['messages'].append({"role": "assistant", "content": response})

    total_tokens = completion.usage.total_tokens
    prompt_tokens = completion.usage.prompt_tokens
    completion_tokens = completion.usage.completion_tokens
    return response, total_tokens, prompt_tokens, completion_tokens

# container for chat history
response_container = st.container()
# container for text box
container = st.container()

#with container:
    # with st.form(key='my_form', clear_on_submit=True):
    #     user_input = st.text_area("You:", key='input', height=50)
    #     submit_button = st.form_submit_button(label='Send')
user_input = get_text()
if user_input:
    output, total_tokens, prompt_tokens, completion_tokens = generate_response(user_input)
    st.session_state['past'].append(user_input)
    st.session_state['generated'].append(output)
    st.session_state['total_tokens'].append(total_tokens)
    st.session_state['mkt_cap'].append(mkt_cap)
    st.session_state['style'].append(style)
    st.session_state['sectors'].append(sectors)
    st.session_state['hedge'].append(hedge)
    st.session_state['longshort'].append(longshort)


if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            #message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            #message(st.session_state["generated"][i], key=str(i))
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            st.write(
                f"Number of tokens: {st.session_state['total_tokens'][i]}.")
        if 'Thank you. I\'ll' in st.session_state['generated'][-1]:
            # End the conversation if "Thank you. I'll" is in the output
            st.session_state['conversation_ended'] = True


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

# Save the conversation and sidebar information as a new file
if 'conversation_ended' in st.session_state:
    if st.session_state['conversation_ended'] == True:
        final_response = st.session_state['generated'][-1]
        triggers = get_node_names(final_response, 'Triggers')
        controls = get_node_names(final_response, 'Controls')
        events = get_node_names(final_response, 'Risk Events')
        mitigators = get_node_names(final_response, 'Mitigators')
        consequences = get_node_names(final_response, 'Consequences')
        edges = get_edges(final_response)

        data = {
            #'past': st.session_state['past'], # user's answers
            'triggers': triggers, 'controls': controls, 
            'events': events, 'mitigators': mitigators, 
            'consequences': consequences, 'edges': edges,
            'mkt_cap': st.session_state['mkt_cap'][-1],
            'style': st.session_state['style'][-1],
            'sectors': st.session_state['sectors'][-1],
            'hedge': st.session_state['hedge'][-1],
            'long/short': st.session_state['longshort'][-1],
        }
        
        with open('E://px//UChi//Courses//Capstone//BN-Creation-Bot//engine//conversation.json', 'w') as file:
            json.dump(data, file)
