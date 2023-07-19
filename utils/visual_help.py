import re, json, openai
import numpy as np
import pandas as pd
import streamlit as st
from utils.utils import get_data

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

def prior_prob_table(node_value, short_name, num_rows):
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

def get_cond_df(priors, num_rows, child_node):
    if child_node == 'MC':
        df = pd.DataFrame({'State': priors['State'].to_list(), 
                        '0-5%': [None] * num_rows,
                        '5-10%': [None] * num_rows,
                        '10-15%': [None] * num_rows,
                        '15-20%': [None] * num_rows,
                        })
        
    elif child_node == 'IL' or 'PL':
        df = pd.DataFrame({'State': priors['State'].to_list(), 
                        '0': [None] * num_rows,
                        '1': [None] * num_rows,
                        })
    
    else:
        df = child_node
    return df


def cond_prob_table(priors, num_rows, short_name, child_node):  
    state_lens = len(priors['State'].to_list())
    df = get_cond_df(priors, num_rows, child_node)
    
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



