from utils.chat_help import *
from utils.utils import get_data
import streamlit as st

# Setting page title and header
st.set_page_config(page_title="BN Chatbot", page_icon=":robot_face:")
st.title("Bayesian Network Creation Chat")
with st.expander("Instructions", expanded=True):
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        ##### **Start a chat:**
        1. Go to sidebar:
            1. <u>Always enter OpenAI API Key first.</u>
            2. Select portfolio info.
        2. Say <u>Hi</u> to ChatGPT and get started!
        3. Click <u>New Chat</u> in the sidebar to restart.
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(
            """
            ##### **After all the info are collected:**
            1. Go to Visualization page for results.
            2. Click the <u>Download</u> button in the sidebar to get a conversation history file. And it will start a new session automatically.
            """,
            unsafe_allow_html=True
        )

template, PROMPT = get_template()

# Initialise session state variables
sessions = ['generated', 'past', 'input', 'stored_session',
            'total_tokens',
            'mkt_cap','style','sectors','hedge','longshort',
            'conversation_ended',
            'input1']

for session in sessions:
    if session not in st.session_state:
        if session != 'input' and session != 'conversation_ended':
            st.session_state[session] = []
        elif session == 'input' or session == 'input1':
            st.session_state[session] = ''
        elif session == 'conversation_ended':
            st.session_state[session] = False   

def submit():
    st.session_state.input1 = st.session_state.input
    st.session_state.input = ''

# Define function to get user input
def get_text():
    input_text = st.text_input("You: ", st.session_state["input"], key="input",
                               on_change=submit,
                               placeholder="Your AI assistant here!",
                               #label_visibility='hidden',
                               )
    #return input_text
    return st.session_state.input1

# Define function to start a new chat
def new_chat():
    """
    Clears session state and starts a new chat.
    """
    save = []
    for i in range(len(st.session_state['generated'])-1, -1, -1):
        save.append("Bot:" + st.session_state["generated"][i])
        save.append("User:" + st.session_state["past"][i])
    
    st.session_state["stored_session"].append(save)
    with open(r'stored_session.txt', 'w') as fp:
        for items in save:
            fp.write("%s\n" % items)

    st.session_state["generated"] = []
    st.session_state["past"] = []
    st.session_state["input"] = ""
    st.session_state["input1"] = ""

    st.session_state['total_tokens'] = []

    st.session_state['mkt_cap'] = []
    st.session_state['style'] = []
    st.session_state['sectors'] = []
    st.session_state['hedge'] = []
    st.session_state['longshort'] = []
    st.session_state['conversation_ended'] = False

    st.session_state.memory.chat_memory.clear()

# Sidebar - API key, portfolio market cap, style, sectors
st.sidebar.title("Info Needed")

# OpenAI API key
API_O = st.sidebar.text_input('OPENAI-API-KEY', type='password')
if API_O:
    llm = OpenAI(temperature=0.3,
                 openai_api_key=API_O,
                #model_name="gpt-3.5-turbo",
                #model_name = 'gpt-3.5-turbo-0613',
                verbose=False)

    if 'memory' not in st.session_state:
        st.session_state.memory = ConversationBufferMemory()

    Conversation = ConversationChain(
            llm=llm, prompt=PROMPT,
            memory = st.session_state.memory
            )

else:
    st.sidebar.warning('API key required.The API key is not stored.')

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


# Add a button to start a new chat
st.sidebar.button("New Chat", on_click = new_chat, type='primary')

# Get the user input
# user_input = get_text()

input_field = st.empty()
with input_field.container():
    user_input = get_text()

# Generate the output using the ConversationChain object and the user input, and add the input/output to the session
#if user_input and st.session_state['conversation_ended'] == False:
if user_input:
    with get_openai_callback() as cb:
        output = Conversation.run(input=user_input)
    st.session_state.past.append(user_input)
    st.session_state.generated.append(output)
    st.session_state.total_tokens.append(cb.total_tokens)
    st.session_state['mkt_cap'].append(mkt_cap)
    st.session_state['style'].append(style)
    st.session_state['sectors'].append(sectors)
    st.session_state['hedge'].append(hedge)
    st.session_state['longshort'].append(longshort)

# Allow to download as well
download_str = []

# Display the conversation history using an expander, and allow the user to download it
with st.expander("Conversation", expanded=True):
    if st.session_state['conversation_ended'] == False:
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            st.success(st.session_state["generated"][i], icon="🤖")
            st.info(st.session_state["past"][i],icon="🧐")
            st.write(f'Token Used: {st.session_state.total_tokens[i]}')

            download_str.append(st.session_state["generated"][i])
            download_str.append(st.session_state["past"][i])

        if len(st.session_state['generated']) > 0 and 'Thank you. I\'ll' in st.session_state['generated'][-1]:
                # End the conversation if "Thank you. I'll" is in the output
                st.session_state['conversation_ended'] = True
                input_field.empty()
                
# Save the conversation and sidebar information as a new file
if 'conversation_ended' in st.session_state and st.session_state['conversation_ended'] == True:
        download_str = '\n'.join(download_str)
        st.sidebar.download_button('Download Conversation & Start New Session', download_str,
                                   file_name = 'conversation_history.txt',
                                   on_click = new_chat,
                                   )
        
        final_response = st.session_state['generated'][-1]
        data = format_data(final_response)

        with open('engine\\conversation.json', 'w') as file:
            json.dump(data, file)
        
        openai.api_key = API_O
        variables = []
        for i in ['triggers','controls','events','mitigators','consequences']:
            if data.get(i):
                variables.extend(list(data.get(i).values()))

        ticker_dic = get_data().economical_ticker_dict
        node_dic = chat_find_tickers(variables, ticker_dic)
        node_dic = eval(node_dic)
        
        with open('engine\\node_dic.json', 'w') as file:
            json.dump(node_dic, file)