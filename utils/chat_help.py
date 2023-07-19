import re, json, openai
import numpy as np
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
Find the ticker values for {} based on the dictionary provided {}.
Return me a python dictionary,
keys are the variables, values are the tickers.
If no ticker found, then put variable:None.
Start the answer with the python dictionary. DO NOT add any more words beside the dictionary.
"""

from utils.utils import get_data
def chat_find_tickers(variables):
    ticker_dic = get_data().economical_ticker_dict
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

# ------------------------------
# For visualization page
def cpd_to_df(cpd):
    variables = cpd.variables
    cardinality = cpd.cardinality
    values = cpd.values.flatten()

    # Generate all combinations of variable states
    index_tuples = pd.MultiIndex.from_product([range(card) for card in cardinality], names=variables)
    df = pd.DataFrame({'Probabilities': values}, index=index_tuples)
    # Rename columns
    for var in variables:
        state_names = cpd.state_names[var]
        state_names = np.array(state_names).astype(str)
        column_name = f'{var} ({", ".join(state_names)})'
        df.rename(columns={var: column_name}, inplace=True)

    # Sort columns
    df = df.reorder_levels(variables, axis=0)
    df.sort_index(axis=0, inplace=True)

    return df

def find_key(dictionary, value):
    for key, val in dictionary.items():
        if val == value:
            return key
    return None

def get_node_fullname():
    """
    get the full names of nodes
    return a dict {'inflation':'INF', 'trade war':'TW', ......}
    """
    
    with open('./engine/conversation.json', 'r') as file:
        json_file = json.load(file)
    new_dict = {}
    for section in json_file:
        if isinstance(json_file[section], dict):
            md_text = "- **{}**: ".format(section.title())
            md_text += "; ".join(["{} ({})".format(value, key) for key, value in json_file[section].items()])
            st.markdown(md_text)
            new_dict.update(json_file[section])
    return new_dict

def get_short_name_by_full(node_value, json_file):
    for node_type in ['triggers', 'controls',
                      'events', 'mitigators',
                      'consequences']:
        if json_file[node_type]:
            if node_type in json_file and node_value in json_file[node_type].values():
                for key, value in json_file[node_type].items():
                    if value == node_value:
                        short_name = key
                        break
    return short_name

def get_child_node(node_value):
    with open('./engine/conversation.json', 'r') as file:
        data = json.load(file)
    
    # Find the short name
    short_name = get_short_name_by_full(node_value, data)
    child_node = None
    for edge in data["edges"]:
        if edge[0] == short_name:
            child_node = edge[1]
            break
    return short_name, child_node

def prior_prob_table(node_value, short_name, num_rows=1):
    num_rows = pd.to_numeric(num_rows)
    df = pd.DataFrame({'State': [None] * num_rows, 
                       'Prob': [None] * num_rows})
    st.data_editor(df, key="prior_table",
                   #num_rows= "dynamic",
                   use_container_width=True, hide_index=True,
                   column_config={
                        'State':st.column_config.Column(help = 'names of states, \
                                                        e.g. for inflation, \
                                                        it can have 3 states: high, normal, low'),
                        'Prob':st.column_config.NumberColumn(
                                                             help='prior probabilities for each state, \
                                                                e.g. for inflation, \
                                                                high/low can be less likely \
                                                                with 10% probability each, \
                                                                while normal is more likely to happen \
                                                                with 80% probability',
                                                             min_value = 0, max_value=1,
                                                             #format = "%d '%'",
                                                             )
                    })

    rows = st.session_state.prior_table['edited_rows']

    # if len(rows) == num_rows:
    #     prob_count = sum('Prob' in row for row in rows.values())
    #     if prob_count == num_rows - 1:
    #         missing_prob = 1 - sum(row['Prob'] for row in rows.values() if 'Prob' in row)
    #         for row in rows.values():
    #             if 'Prob' not in row:
    #                 row['Prob'] = missing_prob
    
    # st.session_state.prior_table['edited_rows'] = rows

    # if num_rows > len(rows):
    #     missing_rows = num_rows - len(rows)
    #     if missing_rows == 1:
    #         st.write
        
    df_save = pd.DataFrame.from_dict(rows, orient='index')
    
    if ('State' and 'Prob' in df_save.columns.values):
        if df_save.Prob.count() == num_rows - 1:
            current_prob = df_save['Prob'].sum()
            left_prob = 1 - current_prob
            df_save['Prob'].fillna(left_prob, inplace=True)
            st.session_state.prior_table['edited_rows']
        # else:
        #     st.write('test')

    if ('State' and 'Prob' in df_save.columns.values):
        prob_sum = df_save['Prob'].sum()
        if prob_sum == 1:
            df_save.to_csv(f'./engine/{short_name}_prior.csv', 
                        index=False)
            return df_save
        else:
            st.warning('The probabilities must sum up to 1!')

def cond_prob_table(priors, num_rows, short_name, child_node):  
    state_lens = len(priors['State'].to_list())
    if child_node == 'MC':
        df = pd.DataFrame({'State': priors['State'].to_list(), 
                        '0-5%': [None] * state_lens,
                        '5-10%': [None] * state_lens,
                        '10-15%': [None] * state_lens,
                        '15-20%': [None] * state_lens,
                        })
        
    elif child_node == 'IL' or 'PL':
        df = pd.DataFrame({'State': priors['State'].to_list(), 
                        '0': [None] * num_rows,
                        '1': [None] * num_rows,
                        })
    
    config = {'State':st.column_config.Column(help = 'names of states based on your specification')}
    for i in df.columns[1:]:
        config_i = {f'{i}':st.column_config.NumberColumn(
                                                    help=f'How likely will this situation {i} happen, \
                                                            depend on each state',
                                                    min_value = 0, max_value=1,
                                                    #format = "%d '%'",
                                                    )}
        config.update(config_i)
    


    st.data_editor(df, key="cond_table",
                    #num_rows= "dynamic",
                    use_container_width=True, hide_index=True,
                    column_config=config
                        )
    
    rows = st.session_state.cond_table['edited_rows']

    df_save = pd.DataFrame.from_dict(rows, orient='index')
    err_flag = False
    for col in df_save.columns:
        col_sum = df_save[col].sum()
        if col_sum != 1:
            err_flag = True
    
    if err_flag == True:
        st.warning('Probabilities for each column should have a sum of 1!')
    else:
        df_save.to_csv(f'./engine/{short_name}_conditional.csv', 
                        index=False)
        return df_save



