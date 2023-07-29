from utils.chat_help import *
from utils.cond_prob_chat import *
import streamlit as st
from streamlit_extras.switch_page_button import switch_page
import re, json, textwrap, openai, os, io
import numpy as np
import pandas as pd
import streamlit as st
from utils.prior_prob_chat import *

def cond_chat_func(input_field1, missing_nodes_name):
    input_field1.empty()
    template = get_cond_probs_template(missing_nodes_name)
    
    sessions = ['generated_3', 'past_3', 'input_3', 'stored_session_3',
                'total_tokens_3', 'messages_3',
                'conversation_ended_3',
                'input1_3']
    for session in sessions:
        if session not in st.session_state:
            if session != 'input_3' and session != 'conversation_ended_3':
                st.session_state[session] = []
            elif session == 'input_3' or session == 'input1_3':
                st.session_state[session] = ''
            elif session == 'conversation_ended_3':
                st.session_state[session] = False

    def submit1():
        st.session_state.input1_3 = st.session_state.input_3
        st.session_state.input_3 = ''

    # Define function to get user input
    def get_text1():
        input_text1 = st.text_input("Conditional Probabilities: ", st.session_state["input_3"], key="input_3",
                                on_change=submit1,
                                placeholder="Your AI assistant here!"
                                )
        #return input_text
        return st.session_state.input1_3

    # Define function to start a new chat
    def new_chat():
        save = []
        for i in range(len(st.session_state['generated_3'])-1, -1, -1):
            save.append("Bot:" + st.session_state["generated_3"][i])
            save.append("User:" + st.session_state["past_3"][i])
        
        st.session_state["stored_session_3"].append(save)
        with open(r'prob_stored_session.txt', 'w') as fp:
            for items in save:
                fp.write("%s\n" % items)

        st.session_state["generated_3"] = []
        st.session_state["past_3"] = []
        st.session_state["input_3"] = ""
        st.session_state["input1_3"] = ""
        st.session_state['total_tokens_3'] = []
        st.session_state['messages_3'] = []
        st.session_state['conversation_ended_3'] = False
        #st.session_state.memory_3.chat_memory.clear()

    #st.sidebar.button("New Chat for Probabilities", on_click = new_chat, type='primary')

    input_field2 = st.empty()
    with input_field2.container():
        user_input1 = get_text1()
        input_field1.empty()

    # generate a response
    def generate_response(prompt):
        st.session_state['messages_3'].append({"role": "user", "content": prompt})
        st.session_state['messages_3'].append({'role':'system', 'content':template})
        
        completion = openai.ChatCompletion.create(
            model="gpt-3.5-turbo-16k-0613",
            messages=st.session_state['messages_3'],
            temperature=0.8,
        )
        
        response = completion.choices[0].message.content
        st.session_state['messages_3'].append({"role": "assistant", "content": response})

        total_tokens = completion.usage.total_tokens
        prompt_tokens = completion.usage.prompt_tokens
        completion_tokens = completion.usage.completion_tokens
        return response, total_tokens, prompt_tokens, completion_tokens
        
    if len(st.session_state.past_3) == 0:
        st.session_state.past_3.append('Hi. How can you help me today?')
        output, total_tokens, prompt_tokens, completion_tokens = generate_response('Hi. How can you help me today?')
        st.session_state.past_3.append(user_input1)
        st.session_state.generated_3.append(output)
        st.session_state.total_tokens_3.append(total_tokens)
    #    st.session_state.total_tokens.append
    else:
        input_field1.empty()
        if user_input1:
            output, total_tokens, prompt_tokens, completion_tokens = generate_response(user_input1)
            input_field1.empty()
            st.session_state.past_3.append(user_input1)
            st.session_state.generated_3.append(output)
            st.session_state.total_tokens_3.append(total_tokens)

    # Allow to download as well
    #download_str = []

    # Display the conversation history using an expander, and allow the user to download it
    with st.expander("Conversation", expanded=True):
        if st.session_state['conversation_ended_3'] == False:
            # iterates through generated index in reverse order
            for i in range(len(st.session_state['generated_3'])-1, -1, -1):
                input_field1.empty()
                st.success(st.session_state["generated_3"][i], icon="🤖")
                if st.session_state["past_3"][i] != 'Hi. How can you help me today?':
                    st.info(st.session_state["past_3"][i+1], icon="🧐")
                st.write(f'Token Used: {st.session_state.total_tokens_3[i]}')

                # download_str.append(st.session_state["generated_3"][i])
                # download_str.append(st.session_state["past_3"][i])

            if len(st.session_state['generated_3']) > 0 and ('Thank you. I\'ll' in st.session_state['generated_3'][-1] or 'I will start' in st.session_state['generated_3'][-1]):
                    # End the conversation if "Thank you. I'll" is in the output
                    st.session_state['conversation_ended_3'] = True
                    # close the input text field
                    input_field1.empty()
                    input_field2.empty()
                    
    # Save conversation & sidebar

    if 'conversation_ended_3' in st.session_state and st.session_state['conversation_ended_3'] == True:
            # download_str = '\n'.join(download_str)
            # st.sidebar.download_button('Download & Restart1', download_str,
            #                         file_name = 'probs_conversation_history.txt',
            #                         on_click = new_chat,
            #                         )
            
            final_response1 = st.session_state['generated_3'][-1]
            x_value = 15
            result_array = create_result_array(final_response1, x_value)
            # st.write(result_array)
            filename = "./engine/cond.npy"
            np.save(filename, result_array)


    #return final_response1


