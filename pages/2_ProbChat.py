from utils.chat_help import *
from utils.cond_prob_chat import *
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import re, json, textwrap, openai, os, io
import numpy as np
import pandas as pd
import streamlit as st
from utils.prior_prob_chat import *

st.set_page_config(page_title="BN Probs", page_icon=":robot_face:")
st.title("Probability Creation Chat")
st.markdown('**Here we collect missing information about probabilities.**')

# get API key
api_path = "./pages/API_O.txt"
file_path = './engine/node_dic.json'

if os.path.exists(api_path):
    with open(api_path, "r") as file:
        API_O = file.read()
        openai.api_key = API_O
else:
    API_O = st.sidebar.text_input('OPENAI-API-KEY', type='password')
    openai.api_key = API_O

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
        #prior_prob_chat(keys_with_null_values)
        
        st.markdown(
                """
                After the conversation is finished:
                <u>Go for Result</u>.
                """,
                unsafe_allow_html=True
            )
        
        go_visual = st.button("Go for Result", "go_visual", type='primary')
        if go_visual:
            switch_page("Visualization")

        missing_nodes_name = keys_with_null_values[0]
        #prob_template = template(missing_nodes_name)
        template = get_prior_probs_template(missing_nodes_name)

        #Initialise session state variables
        sessions = ['generated_2', 'past_2', 'input_2', 'stored_session_2',
                    'total_tokens_2', 'messages',
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
            input_text = st.text_input("Prior Probabilties: ", st.session_state["input_2"], key="input_2",
                                    on_change=submit,
                                    placeholder="Your AI assistant here!"
                                    )
            #return input_text
            return st.session_state.input1_2

        # Define function to start a new chat
        def new_chat():
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
            st.session_state['messages'] = []
            st.session_state['conversation_ended_2'] = False
            #st.session_state.memory_2.chat_memory.clear()

            sessions_to_clear = ['generated_3', 'past_3', 'input_3', 'stored_session_3',
                                'total_tokens_3', 'messages_3',
                                'conversation_ended_3',
                                'input1_3']
            for state_name in sessions_to_clear:
                st.session_state.pop(state_name, None)

            if os.path.exists('./engine/HS_probs.csv'):
                os.remove('./engine/HS_probs.csv')


        st.sidebar.button("New Chat for Probabilities", on_click = new_chat, type='primary')

        input_field1 = st.empty()
        with input_field1.container():
            user_input = get_text()
        
        # generate a response
        def generate_response(prompt):
            st.session_state['messages'].append({"role": "user", "content": prompt})
            st.session_state['messages'].append({'role':'system', 'content':template})
            
            completion = openai.ChatCompletion.create(
                model="gpt-3.5-turbo-16k-0613",
                messages=st.session_state['messages'],
                temperature=0.8,
            )
            
            response = completion.choices[0].message.content
            st.session_state['messages'].append({"role": "assistant", "content": response})

            total_tokens = completion.usage.total_tokens
            prompt_tokens = completion.usage.prompt_tokens
            completion_tokens = completion.usage.completion_tokens
            return response, total_tokens
            
        if len(st.session_state.past_2) == 0:
            first_message = 'Hi. How can you help me today?'
            st.session_state.past_2.append(first_message)
            output, total_tokens = generate_response(first_message)
            # st.session_state.past_2.append(user_input)
            st.session_state.generated_2.append(output)
            st.session_state.total_tokens_2.append(total_tokens)
        #    st.session_state.total_tokens.append
        else:
            if user_input:
                output, total_tokens = generate_response(user_input)
                st.session_state.past_2.append(user_input)
                st.session_state.generated_2.append(output)
                st.session_state.total_tokens_2.append(total_tokens)
        
        # Allow to download as well
        #download_str = []

        conversations = st.empty()
        with conversations.container():
            if st.session_state['conversation_ended_2'] == False:
                # iterates through generated index in reverse order
                for i in range(len(st.session_state['generated_2'])-1, -1, -1):
                    st.success(st.session_state["generated_2"][i], icon="🤖")
                    if st.session_state["past_2"][i] != 'Hi. How can you help me today?':
                        st.info(st.session_state["past_2"][i], icon="🧐")
                    st.write(f'Token Used: {st.session_state.total_tokens_2[i]}')

                    #download_str.append(st.session_state["generated_2"][i])
                    #download_str.append(st.session_state["past_2"][i])

                if len(st.session_state['generated_2']) > 0 and \
                    ('Thank you. I\'ll' in st.session_state['generated_2'][-1] \
                     or 'I will start' in st.session_state['generated_2'][-1]):
                        # End the conversation if "Thank you. I'll" is in the output
                        st.session_state['conversation_ended_2'] = True
                        # close the input text field
                        input_field1.empty()
                        conversations.empty()
                        
        # Save conversation & sidebar
        if 'conversation_ended_2' in st.session_state and st.session_state['conversation_ended_2'] == True:
                #download_str = '\n'.join(download_str)
                #st.sidebar.download_button('Download & Restart1', download_str,
                                        #file_name = 'probs_conversation_history.txt',
                                        #on_click = new_chat,
                                        #)
                
                final_response = st.session_state['generated_2'][-1]
                numeric_values = [float(value) for value in re.findall(r"=\s(\d+\.\d+|\d+)", final_response)]
                if len(numeric_values) > 0:
                    matrix, max_eigenvector_2d = calculate_max_eigenvector(numeric_values)
                    filename = "./engine/prior.npy"
                    np.save(filename, max_eigenvector_2d)
                input_field1.empty()
                


                cond_chat_func(input_field1, missing_nodes_name)