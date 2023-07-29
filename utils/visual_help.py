from engine.Engine import Engine
import re, json, openai
import numpy as np
import pandas as pd
import streamlit as st
from st_aggrid import GridOptionsBuilder, AgGrid, GridUpdateMode, DataReturnMode, JsCode

# ------------------------------
# For visualization page
def cpd_to_df(cpd, node_key, value_meanings):
    variables = cpd.variables
    cardinality = cpd.cardinality
    values = cpd.values.flatten()

    # Generate all combinations of variable states
    index_tuples = pd.MultiIndex.from_product([range(card) for card in cardinality], names=variables)
    df = pd.DataFrame({'Probs': values}, index=index_tuples)
    var_state_names = []
    # Rename columns
    for var in variables:
        state_names = cpd.state_names[var]
        if var != node_key:
            temp_lst = [f"{var}_{value_meanings[var][str(state)]}" for state in state_names]
            var_state_names.extend(temp_lst)
            #st.write(var, state_names)
        
        state_names = np.array(state_names).astype(str)
        
        column_name = f'{var} ({", ".join(state_names)})'
        df.rename(columns={var: column_name}, inplace=True)
    
    # Sort columns
    df = df.reorder_levels(variables, axis=0)
    df.sort_index(axis=0, inplace=True)
    df = df.reset_index()
    df = df.sort_values(by='Probs', ascending=False)

    return df, var_state_names

def get_cpd_agg(df):
    st.markdown('**We used historical data for this node.**')
    node_key = df.columns[0]
    gb = GridOptionsBuilder.from_dataframe(df)
    gb.configure_default_column(groupable=True, value=True, enableRowGroup=True, editable=False)
    gb.configure_column(f"{node_key}", pivot=True)
    gb.configure_column("Probs", type=["numericColumn","numberColumnFilter","customNumericFormat"], precision=3)
    gb.configure_grid_options(domLayout='normal')
    gridOptions = gb.build()
    cpt_table_agg = AgGrid(
            df,  gridOptions=gridOptions,
            height=300,  width='100%',
            columns_auto_size_mode='FIT_ALL_COLUMNS_TO_VIEW',
            editable=False
            )

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

def check_prob(df, short_name, num_rows):
    warning_shown = False
    if not df.notna().all().all():
        st.warning("There're Null values in the table!")
        warning_shown = True
    else:
        try:
            prior_sum = df['Prior Prob'].sum()
            if abs(prior_sum-1) > 1e-3:
                st.warning('Prior probabilities should sum up to 1!')
                warning_shown = True

            prob_cols = df.columns.to_list()[2:]
            for row_index in range(0,num_rows):
                sum_values = df.loc[row_index, prob_cols].sum()
                if abs(sum_values - 1) > 1e-3:
                    st.warning(f'For State {df.State[row_index]}, \
                               the conditional probabilties should sum up to 1!')
                    warning_shown = True
            
            if not warning_shown:
                df.to_csv(f'./engine/{short_name}_probs.csv', index=False)
                    
            
            # prob_sum = df.drop('State', axis=1).sum()
            # if (prob_sum == 1).all():
            #     df.to_csv(f'./engine/{short_name}_probs.csv', index=False)
            # else:
            #     cols_not_sum_to_1 = prob_sum[prob_sum != 1].index.tolist()
            #     st.warning(f"The following columns do not sum up to 1: {cols_not_sum_to_1}")
                
        except:
            st.warning('Probabilities must sum up to 1!')
            warning_shown = True


def build_prob_table(num_rows, child_node, short_name):
    prior = np.load('./engine/prior.npy')
    cond = np.load('./engine/cond.npy')

    cond = pd.DataFrame(cond)
    cond = cond.transpose()
    cond.columns = [f'{child_node}_0', f'{child_node}_1']
    #st.write(cond)

    prior = {'State': ['Hedged', 'Not Hedged'],
            'Prior Prob': prior.flatten()  # Flatten the array to a 1D array for the values
                    }         
    prior = pd.DataFrame(prior)

    df = prior.merge(cond, how='left', left_index=True, right_index=True)

    #n = engine.BN_model.get_cardinality()
    #n = n[child_node]
    # n: cols, b: rows
    # if child_node == 'MC':
    # #n = 4
    #     df[f'{child_node}_0-5%'] = [None] * num_rows
    #     df[f'{child_node}_5-10%'] = [None] * num_rows
    #     df[f'{child_node}_10-15%'] = [None] * num_rows
    #     df[f'{child_node}_15-20%'] = [None] * num_rows

    # else:
    #     df[f'{child_node}_0'] = [None] * num_rows
    #     df[f'{child_node}_1'] = [None] * num_rows
    
    for col in df.columns:
        if col == 'State':
            df[col] = df[col].astype('str')
        else:
            df[col] = df[col].astype('float64')
    df.to_csv(f'./engine/{short_name}_probs.csv', index=False)
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

    df = build_prob_table(num_rows, child_node, short_name=short_name)
    grid_return = AgGrid(df, theme="streamlit", height=200, editable=True,
                         columns_auto_size_mode='FIT_ALL_COLUMNS_TO_VIEW',
                         )

    new_df = grid_return['data']
    check_prob(new_df, short_name, num_rows)  

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
    check_prob(new_df, short_name, num_rows)