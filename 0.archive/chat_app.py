import openai
import streamlit as st
from streamlit_chat import message

# Setting page title and header
st.set_page_config(page_title="BN Bot", page_icon=":robot_face:")
st.markdown("<h1 style='text-align: center;'>Bayesian Network Creation Bot</h1>",
            unsafe_allow_html=True)



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
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
if 'model_name' not in st.session_state:
    st.session_state['model_name'] = []
if 'cost' not in st.session_state:
    st.session_state['cost'] = []
if 'total_tokens' not in st.session_state:
    st.session_state['total_tokens'] = []
if 'total_cost' not in st.session_state:
    st.session_state['total_cost'] = 0.0

# Sidebar - let user choose model, show total cost of current conversation, and let user clear the current conversation
st.sidebar.title("Sidebar")
model_name = st.sidebar.radio("Choose a model:", ("GPT-3.5", "GPT-4"))
counter_placeholder = st.sidebar.empty()
counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")
clear_button = st.sidebar.button("Clear Conversation", key="clear")

# Map model names to OpenAI model IDs
if model_name == "GPT-3.5":
    model = "gpt-3.5-turbo"
else:
    model = "gpt-4"

# Ask the user to enter their OpenAI API key
API_O = st.sidebar.text_input('OPENAI-API-KEY', type='password')
# Session state storage would be ideal
if API_O:
    openai.api_key = API_O
else:
    st.sidebar.warning(
        'API key required to try this app.The API key is not stored in any form.'
    )
    # st.stop()

# reset everything
if clear_button:
    st.session_state['generated'] = []
    st.session_state['past'] = []
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]
    st.session_state['number_tokens'] = []
    st.session_state['model_name'] = []
    st.session_state['cost'] = []
    st.session_state['total_cost'] = 0.0
    st.session_state['total_tokens'] = []
    counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")


# generate a response
def generate_response(prompt):
    st.session_state['messages'].append({"role": "user", "content": prompt})
    st.session_state['messages'].append({'role':'system', 'content':template})
    
    completion = openai.ChatCompletion.create(
        model=model,
        messages=st.session_state['messages'],
        temperature=0.3,
    )
    response = completion.choices[0].message.content
    st.session_state['messages'].append({"role": "assistant", "content": response})

    # print(st.session_state['messages'])
    total_tokens = completion.usage.total_tokens
    prompt_tokens = completion.usage.prompt_tokens
    completion_tokens = completion.usage.completion_tokens
    return response, total_tokens, prompt_tokens, completion_tokens

# container for chat history
response_container = st.container()
# container for text box
container = st.container()

with container:
    with st.form(key='my_form', clear_on_submit=True):
        user_input = st.text_area("You:", key='input', height=100)
        submit_button = st.form_submit_button(label='Send')

    if submit_button and user_input:
        output, total_tokens, prompt_tokens, completion_tokens = generate_response(user_input)
        st.session_state['past'].append(user_input)
        st.session_state['generated'].append(output)
        st.session_state['model_name'].append(model_name)
        st.session_state['total_tokens'].append(total_tokens)

        # from https://openai.com/pricing#language-models
        if model_name == "GPT-3.5":
            cost = total_tokens * 0.002 / 1000
        else:
            cost = (prompt_tokens * 0.03 + completion_tokens * 0.06) / 1000

        st.session_state['cost'].append(cost)
        st.session_state['total_cost'] += cost

if st.session_state['generated']:
    with response_container:
        for i in range(len(st.session_state['generated'])):
            message(st.session_state["past"][i], is_user=True, key=str(i) + '_user')
            message(st.session_state["generated"][i], key=str(i))
            st.write(
                f"Model used: {st.session_state['model_name'][i]}; Number of tokens: {st.session_state['total_tokens'][i]}; Cost: ${st.session_state['cost'][i]:.5f}")
            counter_placeholder.write(f"Total cost of this conversation: ${st.session_state['total_cost']:.5f}")