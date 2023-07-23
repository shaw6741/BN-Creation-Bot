from engine.Engine import Engine
import re, json, openai
import numpy as np
import pandas as pd
import streamlit as st
from st_aggrid import AgGrid
from utils.utils import get_data
from utils.utils import definition_BayesianNetwork as db

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

def get_child_node(node_key):
    with open('./engine/conversation.json', 'r') as file:
        data = json.load(file)
    child_node = None
    for edge in data["edges"]:
        if edge[0] == node_key:
            child_node = edge[1]
            break
    return child_node

def json_to_df(json_data):
    return pd.DataFrame.from_dict(json_data, orient='index')

def df_to_json(dataframe):
    return dataframe.to_dict(orient='index')

def check_prob(df, short_name):
    if not df.notna().all().all():
        st.warning("There're Null values in the table!")
    else:
        try:
            prob_sum = df.drop('State', axis=1).sum()
            if (prob_sum == 1).all():
                df.to_csv(f'./engine/{short_name}_probs.csv', index=False)
            else:
                cols_not_sum_to_1 = prob_sum[prob_sum != 1].index.tolist()
                st.warning(f"The following columns do not sum up to 1: {cols_not_sum_to_1}")
                
        except:
            st.warning('Probabilities must sum up to 1!')


def build_prob_table(num_rows, child_node):
    df = pd.DataFrame(
        {
            'State':[None]*num_rows,
            'Prior Prob':[None]*num_rows,
        }
    )
    #n = engine.BN_model.get_cardinality()
    #n = n[child_node]
    # n: cols, b: rows
    if child_node == 'MC':
    #n = 4
        df[f'{child_node}_0-5%'] = [None] * num_rows
        df[f'{child_node}_5-10%'] = [None] * num_rows
        df[f'{child_node}_10-15%'] = [None] * num_rows
        df[f'{child_node}_15-20%'] = [None] * num_rows

    else:
        df[f'{child_node}_0'] = [None] * num_rows
        df[f'{child_node}_1'] = [None] * num_rows
    
    for col in df.columns:
        if col == 'State':
            df[col] = df[col].astype('str')
        else:
            df[col] = df[col].astype('float64')

    return df

def update_num_states(num_rows, df):
    if num_rows != df.shape[0]:
        # If the number of rows is less, add rows with None
        if num_rows > df.shape[0]:
            diff_rows = num_rows - df.shape[0]

            concat_df = pd.DataFrame({'State': [None] * diff_rows})
            for col in df.columns:
                concat_df[f'{col}'] = [None] * diff_rows
            
            df = pd.concat([df, concat_df], ignore_index=True)

        # If the number of rows is more, delete excess rows
        else:
            df = df.iloc[:num_rows]
            
    for col in df.columns:
        if col == 'State':
            df[col] = df[col].astype('str')
        else:
            df[col] = df[col].astype('float64')
    
    return df

def get_prob_table0(short_name, child_node):
    number = st.slider('Number of States', 1, 5, value=2)
    num_rows = pd.to_numeric(number)

    df = build_prob_table(num_rows, child_node)
    grid_return = AgGrid(df, theme="streamlit", height=200, editable=True,
                         columns_auto_size_mode='FIT_ALL_COLUMNS_TO_VIEW',
                         )

    new_df = grid_return['data']
    check_prob(new_df, short_name)  

def get_prob_table1(short_name):
    last_edited = pd.read_csv(f'./engine/{short_name}_probs.csv', index_col=False)
    #last_edited = json_to_df(st.session_state[f'{short_name}_probs'][-1])
    number = st.slider('Number of states', 1, 5, value=last_edited.shape[0])
    num_rows = pd.to_numeric(number)

    df = last_edited.copy()
    df = update_num_states(num_rows, df)    

    grid_return = AgGrid(df, theme="streamlit", height=200, editable=True,
                         fit_columns_on_grid_load=True,
                         )
    
    new_df = grid_return['data']
    check_prob(new_df, short_name)