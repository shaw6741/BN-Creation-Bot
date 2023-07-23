from utils.chat_help import *
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import re, json, textwrap, openai, os, io
import numpy as np
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory, ChatMessageHistory
from langchain.prompts.prompt import PromptTemplate
from langchain.schema import messages_from_dict, messages_to_dict
from langchain.callbacks import get_openai_callback
import pandas as pd
import streamlit as st

st.set_page_config(page_title="BN Probs", page_icon=":robot_face:")
st.title("Probability Creation Chat")
st.markdown('**Here we collect missing information about probabilities.**')

# get API key
api_path = "./pages/API_O.txt"
file_path = './engine/node_dic.json'

if os.path.exists(api_path):
    with open(api_path, "r") as file:
        API_O = file.read()
else:
    API_O = st.sidebar.text_input('OPENAI-API-KEY', type='password')

# get missing nodes

if os.path.exists(file_path):
    with open(file_path, 'r') as file:
        node_dic = json.load(file)
    keys_with_null_values = [key for key, value in node_dic.items() if value is None]
else:
    st.warning('Please go to Chat Page to finishing collecting required information')


if 'keys_with_null_values' in globals():
    if len(keys_with_null_values) == 0:
        st.markdown(
                """
                We have historical data for all the nodes you mentioned before!
                <u>Go for Result</u>.
                """,
                unsafe_allow_html=True
            )
        go_visual = st.button("Go for Result", "go_visual", type='primary')
        if go_visual:
            switch_page("Visualization")

    else:
        missing_nodes_name = keys_with_null_values[0]
        prob_template, prob_prompt = get_probs_template(missing_nodes_name)

        # Initialise session state variables
        sessions = ['generated_2', 'past_2', 'input_2', 'stored_session_2',
                    'total_tokens_2',
                    'conversation_ended_2',
                    'input1_2']

        for session in sessions:
            if session not in st.session_state:
                if session != 'input_2' and session != 'conversation_ended_2':
                    st.session_state[session] = []
                elif session == 'input_2' or session == 'input1_2':
                    st.session_state[session] = ''
                elif session == 'conversation_ended_2':
                    st.session_state[session] = False   

        def submit():
            st.session_state.input1_2 = st.session_state.input_2
            st.session_state.input_2 = ''

        # Define function to get user input
        def get_text():
            input_text = st.text_input("You: ", st.session_state["input_2"], key="input_2",
                                    on_change=submit,
                                    placeholder="Your AI assistant here!"
                                    )
            #return input_text
            return st.session_state.input1_2

        # Define function to start a new chat
        def new_chat():
            """
            Clears session state and starts a new chat.
            """
            save = []
            for i in range(len(st.session_state['generated_2'])-1, -1, -1):
                save.append("Bot:" + st.session_state["generated_2"][i])
                save.append("User:" + st.session_state["past_2"][i])
            
            st.session_state["stored_session_2"].append(save)
            with open(r'prob_stored_session.txt', 'w') as fp:
                for items in save:
                    fp.write("%s\n" % items)

            st.session_state["generated_2"] = []
            st.session_state["past_2"] = []
            st.session_state["input_2"] = ""
            st.session_state["input1_2"] = ""
            st.session_state['total_tokens_2'] = []
            st.session_state['conversation_ended_2'] = False
            st.session_state.memory_2.chat_memory.clear()



        if API_O:
            llm = OpenAI(
                        temperature=0.5,
                        openai_api_key=API_O,
                        #model_name="gpt-3.5-turbo",
                        #model_name = 'gpt-3.5-turbo-0613',
                        model_name = 'gpt-3.5-turbo-16k-0613',
                        verbose=True)

            if 'memory_2' not in st.session_state:
                #st.session_state.memory = ConversationBufferMemory()
                st.session_state.memory_2 = ConversationSummaryMemory(llm=llm)
            
            Conversation = ConversationChain(
                    llm=llm, prompt=prob_prompt,
                    memory = st.session_state.memory_2,
                    verbose= False,
                    )
    
        # Add a button to start a new chat
        st.sidebar.button("New Chat1", on_click = new_chat, type='primary')

        # get user input
        # create container to auto close the input field after reaching the summary from gpt
        input_field1 = st.empty()
        with input_field1.container():
            user_input = get_text()

        #if user_input and st.session_state['conversation_ended'] == False:
        if API_O and len(st.session_state.past_2) == 0:
            st.session_state.past_2.append('Hi. How can you help me today?')
            with get_openai_callback() as cb:
                output = Conversation.run(input = st.session_state.past_2[0])
            
            st.session_state.generated_2.append(output)
            st.session_state.total_tokens_2.append(cb.total_tokens)
        #    st.session_state.total_tokens.append
        else:
            if API_O and user_input:
                with get_openai_callback() as cb:
                    output = Conversation.run(input=user_input)
                st.session_state.past_2.append(user_input)
                st.session_state.generated_2.append(output)
                st.session_state.total_tokens_2.append(cb.total_tokens)

        # Allow to download as well
        download_str = []

        # Display the conversation history using an expander, and allow the user to download it
        with st.expander("Conversation", expanded=True):
            if st.session_state['conversation_ended_2'] == False:
                # iterates through generated index in reverse order
                for i in range(len(st.session_state['generated_2'])-1, -1, -1):
                    st.success(st.session_state["generated_2"][i], icon="🤖")
                    if st.session_state["past_2"][i] != 'Hi. How can you help me today?':
                        st.info(st.session_state["past_2"][i], icon="🧐")
                    st.write(f'Token Used: {st.session_state.total_tokens_2[i]}')

                    download_str.append(st.session_state["generated_2"][i])
                    download_str.append(st.session_state["past_2"][i])

                if len(st.session_state['generated_2']) > 0 and 'Thank you. I\'ll' in st.session_state['generated_2'][-1]:
                        # End the conversation if "Thank you. I'll" is in the output
                        st.session_state['conversation_ended_2'] = True
                        # close the input text field
                        input_field1.empty()
                        
        # Save conversation & sidebar

        if 'conversation_ended_2' in st.session_state and st.session_state['conversation_ended_2'] == True:
                download_str = '\n'.join(download_str)
                st.sidebar.download_button('Download & Restart1', download_str,
                                        file_name = 'probs_conversation_history.txt',
                                        on_click = new_chat,
                                        )
                
                final_response = st.session_state['generated_2'][-1]
                
                # Define the pattern for extracting the names and values
                pattern = r'=\s(\d+\.\d+|\d+)'
                matches = re.findall(pattern, final_response)
                values = [float(value) for value in matches]
                st.write(values)
                new_chat()
                #matrices, CR_values = calculate_max_eigenvector(values)
                
                # # save conversation file
                # data = format_data(final_response)
                # with open('engine\\probs_conversation.json', 'w') as file:
                #     json.dump(data, file)
                
                # # save json to find tickers
                # openai.api_key = API_O
                # variables = []
                # for i in ['triggers','controls','events','mitigators','consequences']:
                #     if data.get(i):
                #         variables.extend(list(data.get(i).values()))

                # node_dic = chat_find_tickers(variables)
                # node_dic = eval(node_dic)
                
                # with open('engine\\node_dic.json', 'w') as file:
                #     json.dump(node_dic, file)