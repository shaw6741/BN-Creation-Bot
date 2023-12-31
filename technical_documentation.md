# **main.py**
A Python script that runs a command using the Streamlit library. **Only executed when the script is run directly and not when it is imported as a module.**

`run_command()`: Function to run the Streamlit application `About.py`.
    This function uses the 'os.system' method to execute the 'streamlit run About.py' command,
    which runs the Streamlit application.

`main()`: Main function to start the engine and run the Streamlit application.
    This function logs the message `'Starting Engine'` using `logging.info` method and then
    calls the `run_command` function to execute the Streamlit application.

# **About.py**
Thie code is a Streamlit application that creates a home page for a Bayesian Network creation app.

# **utils.py**
A set of classes and functions for data manipulation, Bayesian network modeling, and visualization. 
It allows users to fetch historical data, calculate rolling returns, perform Monte Carlo corrections, simulate data based on a Bayesian network model, 
and visualize the network structure.


`yf.pdr_override()`: Overrides the pandas_datareader module with the yfinance module (`pandas_datareader`: library used to etract financial data from various sources)

## ***class* `data_utilities`**

`get_busines_day(self, BusDate)`: Checks if the given date is a Saturday or Sunday and adjusts it accordingly.

`get_start_end_dates(self, time_horizon=5)`: The code calculates the start and end dates for a given time horizon (in 5 years), 
                                            taking into account weekends and adjusting the dates to the previous business days if necessary. 
                                            The start date is set to a specific number of years before the end date, providing a time window for analysis or calculations.

`get_rolling_returns(self, df, periods=1, roll_window=5)` : Calculates the rolling returns of a given df over 1 period using a rolling window of size roll_window (5).

`get_mc_correction(self, df, col_name='MC')` : Calculates the Market Correction(MC) based on the values in the DataFrame df. 
                                             It checks if the values fall into certain ranges and assigns corresponding correction levels (0, 1, 2, or 3). 
                                             " 0: not mc, 1: mc of 5%, 2 mc of 10% and 3 mc of 20% "

`get_portfolio_loss(self, df, port_loss, col_name, side, hedged)` : Calculates the portfolio loss based on the values in the DataFrame df and the given parameters. 
                                                                  It applies a loss percentage (port_loss) to the values in df based on the side and hedged flags. 


## ***class* `get_data(data_utilities)`**

Provides a set of methods to fetch historical data from different sources (Yahoo Finance, FRED) based on ticker symbols or portfolio names, 
and perform calculations such as creating mixed indices for stocks and retrieving portfolio data.

`__init__(self)` : Initializes the class and sets up the fred attribute with an API key for accessing FRED (Federal Reserve Economic Data) 
                 and defines dictionaries for mapping ticker symbols to data sources (market or economical)

`get_historical_data(self, ticker='SP', start_date='1970-01-01', end_date='2023-06-01')` : retrieves historical data for a given ticker (symbol), start_date, and end_date

`get_yahoo(self, ticker, start_date, end_date)` : Retrieves historical stock data from Yahoo Finance for a given ticker, start_date, and end_date.

`get_fred(self, ticker, start_date, end_date)` :  retrieves historical economic data from FRED for a given ticker, start_date, and end_date.

`create_mixed_index(self, stockA_df, stockB_df, p_a=0.5, p_b=0.5, inv=10000.0)` : Creates a mixed index by combining the data from two stocks 
                                                                                (stockA_df and stockB_df) based on given weights (p_a and p_b).

`get_df_portfolio(self, client_portfolio, start, end)` : Retrieves historical data for a given client_portfolio (Core, ) 
                                                       and time period (start and end).


## ***class* `call_arviz_lib`**
Provide convenient wrappers around various visualization and analysis functions from the ArviZ library, 
allowing users to easily access and explore the results of Bayesian models

`get_summary(self, data, round_to=2)` : Takes a data object representing Bayesian model samples and returns a summary table of 
                                      posterior statistics using the az.summary function from ArviZ

`get_plot_trace(self, data, compact=False)` : Takes a data object and plots the trace of the parameters in the Bayesian model using the az.plot_trace function from ArviZ

`get_autocorrelation(self, data, max_lag=30, combined=True)` : Takes a data object and plots the autocorrelation of the parameters in the Bayesian model 
                                                             using the az.plot_autocorr function from ArviZ. 

`get_plot_ess(self, data, kind)` : Takes a data object and plots the effective sample size (ESS) of the parameters in the Bayesian model using 
                                 the az.plot_ess function from ArviZ.

`get_plot_forest(self, data, hdi_prob=0.95, r_hat=True, ess=True)` : Method takes a data object and plots a forest plot of the parameters in the Bayesian model using the az.plot_forest function from ArviZ. 
                                                                   The hdi_prob parameter determines the probability mass of the highest density interval (HDI) to display, and the r_hat and 
                                                                   ess parameters control whether to display the R-hat statistic and ESS values, respectively.

`get_ess(self, data, var_names)` : Takes a data object and a list of var_names (parameter names) and returns the effective sample size (ESS) for the specified parameters 
                                 using the az.ess function from ArviZ.

`get_rhat(self, data, var_names)` : Takes a data object and a list of var_names (parameter names) and returns the R-hat statistic for the specified parameters using the az.rhat function from ArviZ.

`get_plot_mcse(self, data, extra_methods=True)` : Takes a data object and plots the Monte Carlo standard error (MCSE) of the parameters in the Bayesian model using the az.plot_mcse function from ArviZ. 
                                                The extra_methods parameter controls whether to include additional estimation methods in the plot.

`get_posterior(self, data)`:  Takes a data object and plots the posterior distribution of the parameters in the Bayesian model using the az.plot_posterior function from ArviZ

`get_dist(self, data)` : Takes a data object and plots the marginal posterior distribution of the parameters in the Bayesian model using the az.plot_dist function 


## ***class* `definition_BayesianNetwork`**

`create_network(self, nodes)` : Initializes a Bayesian network with the given nodes.

`create_simple_cpd(self, node_name, num_cards, vals)`: BayesianNetwork(nodes) : Creates a conditional probability distribution (CPD) for a single node with node_name, num_cards (number of possible states), 
                                                                              and vals (probability values)

`create_output_cpd(self, node_name, num_cards, vals, evidence, evid_cards)`: Creates an output CPD for a node with node_name, num_cards, vals (probability values), evidence (evidence nodes), 
                                                                           and evid_cards (number of possible states for evidence nodes).

`fit_network(self, df)`: Fits the Bayesian network to the given df (data) using maximum likelihood estimation.

`fit_update_network(self, df)`: Updates the Bayesian network with additional data from df.

`get_predict(self, predict_data)`: Performs inference and predicts the most probable states for the given predict_data.

`get_predict_probability(self, predict_data)`: Performs inference and returns the probability distribution over states for the given predict_data.

`model_add_cpds(self, cpd)`: Adds the given CPD to the Bayesian network.

`model_remove_cpds(self, cpd)`: Removes the given CPD from the Bayesian network.

`inference_object(self, var_name, evidence)`: Creates an inference object using variable elimination for the given var_name (target variable) and evidence (evidence nodes) and 
                                            returns the posterior probabilities.

`check_model(self)`: Checks the model for validity and returns a histogram of the model.

`model_simulate(self, samples)`: Generates samples from the Bayesian network.

`get_nodes(self)`: Returns the nodes in the Bayesian network.

`get_edges(self)`: Returns the edges (dependencies) in the Bayesian network.

`get_cpds(self)`: Returns the CPDs in the Bayesian network.

`get_cardinality(self)`: Returns the cardinality (number of possible states) of the variables in the Bayesian network.

`get_mle(self, data)`: Performs maximum likelihood estimation on the given data to estimate the parameters of the Bayesian network and returns the Maximum Likelihood Estimator object.

`get_bic_est(self, data)`: Computes the Bayesian Information Criterion (BIC) score for the given data and the Bayesian network.

`get_k2_est(self, data)`: Computes the K2 score for the given data and the Bayesian network.

`get_log_l_score(self, data)`: Computes the log-likelihood score of the given data for the Bayesian network.

`to_mm_model(self)`: Converts the Bayesian network to a Markov model.

`plot_daft(self)`: Renders a Daft-style plot of the Bayesian network using the daft library.

`plot_networkx(self)`: Creates a networkx plot of the Bayesian network using the networkx library. 
                     The nodes are placed in layers based on their topological ordering, and the plot is returned as a matplotlib figure object.


`get_hist_data_from_BN(raw_data)`: Utilizes a trained Bayesian network to generate simulated data based on the provided raw_data.

# **bayesian_model.py:**
This code defines a class model_run that is used to run a Bayesian Network model. 

`class model_run()`: Defines a class model_run that inherits from definition_BayesianNetwork
_init__ : is the class constructor that initializes various attributes used by the Bayesian Network model

`get_indicators_node_list(self)`: This method populates two lists: indicators_node_list and missing_node_list.
                                 Iterates over the nodes and checks if each node is an indicator node (found in market ticker or economical ticker dictionaries) 
                                 / a missing node
                                
`construct_returns(self, ticker)`: Constructs the returns for a given ticker by retrieving the close price data using get_df_portfolio() 
                                 and then calculating the returns and rolling returns using get_rolling_returns().

`construct_mkt_crash(self)` : Constructs the market crash data by calling construct_returns() for the market ticker. 
                            It then retrieves the market correction data using get_mc_correction(). 
                            The resulting dataframe is stored in self.mc_df, and the event node name is added to the not_indicator_node_list.

`construct_portfolio_loss(self)` : Constructs the portfolio loss data by calling construct_returns() for the client portfolio. 
                                 It then retrieves the portfolio loss data using get_portfolio_loss(). 
                                 The resulting dataframe is stored in self.port_c_df, and the consequence node name is added to the not_indicator_node_list.

`merge_df(self)` : Merges the market crash data (self.mc_df), economic data (self.eco_data), and portfolio loss data (self.port_c_df)
                 into a single dataframe (self.merged_data). 
                 The missing values are filled using forward fill method, and any remaining rows with missing values are dropped.

`generate_bin_data(self)` : This method generates binary data for the indicator nodes in the Bayesian Network. 
                          It creates a deep copy of self.merged_data as self.bn_data and applies binary transformations 
                          based on threshold values (self.oil_jump and self.inf_jump) for oil and inflation indicator nodes.

`get_rnd(self, n, pct_0=60)` : This method generates a random array of size n with a given percentage of zeros (pct_0). 
                             Zeros and ones are chosen with probabilities calculated based on the given percentage.

`generate_missing_data(self)` :  It iterates over the missing_node_list and generates random data arrays using get_rnd() based on the type of missing node. 
                               For mitigators, it generates random binary data with all ones. 
                               For unique events, it retrieves historical data using get_hist_data_from_BN(). 
                               For other missing nodes, it generates random binary data with a random percentage of zeros.

`initialized_data(self)` : This method initializes the data required for the Bayesian Network model. 
                         It calls the necessary methods to construct the market crash data, portfolio loss data, and indicator node lists. 
                         It retrieves historical data for indicator nodes and concatenates them to self.eco_data. 
                         It then merges the data, generates binary data, and generates missing data.

`run_BN_model(self)` : Runs the Bayesian Network model. 
                     It calls initialized_data() to prepare the data, then calls create_network() and fit_network() methods 
                     from the definition_BayesianNetwork class to create and fit the network using the prepared data.

# **Engine.py**
This code provide a way to read JSON data from files, populate attributes with the data, 
and start the Bayesian network model by creating an instance of the model_run class with the necessary inputs.

## ***class* `read_json`** 
`__init__` : Initializes various attributes such as file paths (path_conversation and path_chatgpt_dict), data, and chatgpt_node_dict. 
           It also sets default values for several inputs related to the Bayesian network.

`read_` : Defined to read JSON data from the path_conversation file and load it into the data attribute. 
        It also reads JSON data from the path_chatgpt_dict file and loads it into the chatgpt_node_dict attribute.
        Additionally, it filters out any key-value pairs in chatgpt_node_dict where the value is None.

`filter_empty_keys` : Checks for specific keys in the data attribute and sets corresponding attributes with the same name as the key if the value is not None.


## ***class* `Engine`** 
`__init__` : First calls the read_ method of the read_json class to populate the attributes with JSON data.

`read_` : Further populates various attributes specific to the Engine class using the data from the data attribute.

`start` : Initializes an instance of the model_run class (not shown in the provided code) with the attributes and values from the Engine class. 
        It then calls the run_BN_model method of the BN_model instance.



