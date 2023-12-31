import numpy as np
import pandas as pd
import copy
import datetime
import random
import statsmodels.api as sm
from pgmpy.factors.discrete.CPD import TabularCPD
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from utils.utils import get_data, definition_BayesianNetwork, get_hist_data_from_BN


class model_run(definition_BayesianNetwork):
    def __init__(self, mkt_ticker, client_portfolio, nodes, side_, hedged_, portfolio_loss, oil_jump, inf_jump,
                 fed_hike, diff_periods, time_horizon, roll_win, triggers_dict, controls_dict, events_dict,
                 mitigators_dict, consequences_dict, translator_dict, prior_prob, conditional_prob):
        self.mkt_ticker = mkt_ticker
        self.client_portfolio = client_portfolio
        self.nodes = nodes
        self.portfolio_side = side_
        self.portfolio_hedged = hedged_
        self.portfolio_loss = portfolio_loss
        self.oil_jump = oil_jump
        self.inf_jump = inf_jump
        self.fed_hike = fed_hike
        self.diff_periods = diff_periods
        self.time_horizon = time_horizon
        self.roll_win = roll_win
        self.translate_node_tick = translator_dict
        self.prior_prob = prior_prob
        self.conditional_prob = conditional_prob

        if triggers_dict:
            self.triggers_list = list(triggers_dict.keys())
        if controls_dict:
            self.controls_list = list(controls_dict.keys())
        if events_dict:
            self.events_list = list(events_dict.keys())
        if mitigators_dict:
            self.mitigators_list = list(mitigators_dict.keys())
            self.portfolio_hedged = True
        if consequences_dict:
            self.consequences_list = list(consequences_dict.keys())

        self.indicators_node_list = []
        self.missing_node_list = []
        self.not_indicator_node_list = []

        self.oil_syn = ['CRUDE', 'OIL', 'OSD']
        self.inf_syn = ['INF', 'CPI', 'IN']
        self.fed_syn = ['FED', 'FR', 'FRI']
        self.unemployment_syn = ['UNP', 'Unemployment']
        self.unique_events = ['TW', 'DW']

        self.data_pull = get_data()
        self.start, self.end = self.data_pull.get_start_end_dates(self.time_horizon)

        self.value_meanings = {}

    def get_indicators_node_list(self):
        for node_list in self.nodes:
            for node in node_list:
                if node in self.not_indicator_node_list:
                    next
                elif node in self.data_pull.market_ticker_dict.keys() or node in self.data_pull.economical_ticker_dict.keys():
                    if node not in self.indicators_node_list:
                        self.indicators_node_list.append(node)
                else:
                    if node not in self.missing_node_list:
                        self.missing_node_list.append(node)

    def get_portfolio_returns(self, port_ticker):
        df_close = self.data_pull.get_df_portfolio(port_ticker, self.start, self.end)
        df_ret, df_roll_ret = self.data_pull.get_rolling_returns(df_close, self.diff_periods, self.roll_win)
        return df_ret, df_roll_ret

    def get_ticker_returns(self, ticker):
        if ticker in self.translate_node_tick.keys():
            df_close = self.data_pull.get_historical_data(ticker, self.start, self.end, self.translate_node_tick[ticker])
        else:
            df_close = self.data_pull.get_historical_data(ticker, self.start, self.end)
        df_ret, df_roll_ret = self.data_pull.get_rolling_returns(df_close, self.diff_periods, self.roll_win)
        return df_ret, df_roll_ret

    def construct_mkt_crash(self):
        _, mc_roll_ret = self.get_portfolio_returns(self.mkt_ticker)
        if self.events_list:
            self.mc_df, mc_value_meanings = self.data_pull.get_mc_correction(mc_roll_ret, self.events_list[0])
            self.value_meanings.update(mc_value_meanings)
            self.not_indicator_node_list.append(self.events_list[0])
        else:
            self.mc_df, mc_value_meanings = self.data_pull.get_mc_correction(mc_roll_ret, 'MC')
            self.value_meanings.update(mc_value_meanings)
            self.not_indicator_node_list.append('MC')

    def construct_portfolio_loss(self):
        _, port_roll_ret = self.get_portfolio_returns(self.client_portfolio)
        if self.consequences_list:
            self.port_c_df, loss_value_meanings = self.data_pull.get_portfolio_loss(port_roll_ret, self.portfolio_loss, self.consequences_list[0], self.portfolio_side)
            self.value_meanings.update(loss_value_meanings)
            self.not_indicator_node_list.append(self.consequences_list[0])
        else:
            self.port_c_df, loss_value_meanings = self.data_pull.get_portfolio_loss(port_roll_ret, self.portfolio_loss, 'PIL', self.portfolio_side)
            self.value_meanings.update(loss_value_meanings)
            self.not_indicator_node_list.append('PIL')

    def merge_df(self):
        self.merged_data = pd.concat([self.mc_df, self.eco_data, self.port_c_df], axis=1)
        self.merged_data.fillna(0, inplace=True)
        # self.merged_data.fillna(method='ffill', inplace=True)
        # self.merged_data.dropna(inplace=True)

    def generate_bin_data(self):
        self.bn_data = copy.deepcopy(self.merged_data)
        for ind in self.indicators_node_list:
            if ind in self.oil_syn:
                self.bn_data[ind] = (self.bn_data[ind] >= self.oil_jump) | (self.bn_data[ind] <= -self.oil_jump)
                ind_value_meanings = {
                    ind: {'1': f'1:x>={self.oil_jump} & x<=-{self.oil_jump}',
                          '0': f'0:-{self.oil_jump}<x<{self.oil_jump}'}
                }
                self.value_meanings.update(ind_value_meanings)
            elif ind in self.inf_syn:
                self.bn_data[ind] = self.bn_data[ind] >= self.inf_jump
                ind_value_meanings = {
                    ind: {'1': f'1:change>={self.inf_jump}',
                          '0': f'0:change<{self.inf_jump}'}
                }
                self.value_meanings.update(ind_value_meanings)
            elif ind in self.fed_syn:
                self.bn_data[ind] = self.bn_data[ind] >= self.fed_hike
                ind_value_meanings = {
                    ind: {'1': f'1:change>={self.fed_hike}',
                          '0': f'0:change<{self.fed_hike}'}
                }
                self.value_meanings.update(ind_value_meanings)
            else:
                self.bn_data[ind] = self.bn_data[ind] >= 0
                ind_value_meanings = {
                    ind: {'1': '1:change>=0',
                          '0': '0:change<0'}
                }
                self.value_meanings.update(ind_value_meanings)
            self.bn_data[ind] = self.bn_data[ind].astype(int)

    def get_rnd(self, n, pct_0=60):
        random_array = np.random.choice([0, 1], size=n, p=[pct_0 / 100, (100 - pct_0) / 100])
        return random_array

    def generate_missing_data(self):
        # get some random date for other parameters
        N = len(self.bn_data)
        for col_name in self.missing_node_list:
            if col_name in self.mitigators_list:
                temp_array = self.get_rnd(N, 100)
            elif col_name in self.unique_events:
                if self.events_list:
                    temp_array, war_value_meanings = get_hist_data_from_BN(self.bn_data[self.events_list[0]])
                    self.value_meanings.update(war_value_meanings)
                else:
                    temp_array, war_value_meanings = get_hist_data_from_BN(self.bn_data['MC'])
                    self.value_meanings.update(war_value_meanings)
            else:
                rand_num = random.randint(50, 99)
                temp_array = self.get_rnd(N, rand_num)
            self.bn_data[col_name] = temp_array

    def initialized_data(self):
        self.construct_mkt_crash()
        self.construct_portfolio_loss()
        self.get_indicators_node_list()

        counter = 0
        for ind in self.indicators_node_list:
            ind_ret, int_roll_ret = self.get_ticker_returns(ind)
            if (ind in self.inf_syn) or (ind in self.fed_syn):
                temp_df = copy.deepcopy(ind_ret)
            else:
                temp_df = copy.deepcopy(int_roll_ret)

            if counter == 0:
                self.eco_data = copy.deepcopy(temp_df)
            else:
                self.eco_data = pd.concat([self.eco_data, temp_df], axis=1)
            counter += 1
        self.eco_data.columns = self.indicators_node_list
        self.eco_data.fillna(0, inplace=True)

        self.merge_df()
        self.generate_bin_data()
        self.generate_missing_data()

    def update_network_hedge(self):
        def mix_prob(prev, new):
            not_hedge_ = new[0][0]
            hedge_ = new[0][1]

            inv_win_list = []
            inv_loss_list = []

            for i in range(len(prev[0][0])):
                win_val = float(prev[0][0][i] * not_hedge_)
                loss_val = 1.0 - win_val
                inv_win_list.append(win_val)
                inv_loss_list.append(loss_val)

            for i in range(len(prev[0][0])):
                loss_val = float(prev_prob[1][0][i] * hedge_)
                win_val = 1.0 - loss_val
                inv_win_list.append(win_val)
                inv_loss_list.append(loss_val)

            return np.array([inv_win_list, inv_loss_list])

        node = self.mitigators_list[0]

        prior_prob = np.array([list(self.prior_prob[1]), list(self.prior_prob[0])])
        prior_cpd = TabularCPD(node, 2, prior_prob)

        # cond_prob = np.array([list(self.conditional_prob[1]), list(self.conditional_prob[0])])
        cond_prob = np.array([list(self.conditional_prob[0]), list(self.conditional_prob[1])])
        if self.consequences_list:
            prev_prob = self.model.get_cpds(self.consequences_list[0]).values
            # new_cond_prob = mix_prob(prev_prob, cond_prob)
            new_cond_prob = cond_prob
            if self.events_list:
                cond_cpd = TabularCPD(self.consequences_list[0], 2, new_cond_prob, evidence=[node, self.events_list[0]], evidence_card=[2, 4])
            else:
                cond_cpd = TabularCPD(self.consequences_list[0], 2, new_cond_prob, evidence=[node, 'MC'], evidence_card=[2, 4])
        else:
            prev_prob = self.model.get_cpds('PIL').values
            # new_cond_prob = mix_prob(prev_prob, cond_prob)
            new_cond_prob = cond_prob
            if self.events_list:
                cond_cpd = TabularCPD('PIL', 2, new_cond_prob, evidence=[node, self.events_list[0]], evidence_card=[2, 4])
            else:
                cond_cpd = TabularCPD('PIL', 2, new_cond_prob, evidence=[node, 'MC'], evidence_card=[2, 4])
        
        self.model.add_cpds(prior_cpd, cond_cpd)
        
    def get_regression_data(self, train_size=0.5):
        df = copy.deepcopy(self.bn_data)
        if self.consequences_list[0]:
            X = df.drop([self.consequences_list[0]], axis=1)
            y = df[self.consequences_list[0]]
        else:
            X = df.drop(['PIL'], axis=1)
            y = df['PIL']

        # X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_size, random_state=0)

        reg_model = sm.OLS(y, X.assign(const=1))
        reg_results = reg_model.fit()
        self.reg_predictions = reg_results.predict()

        model = RandomForestClassifier()
        model.fit(X, y)
        self.rf_predictions = model.predict(X)

        self.bn_predictions = self.model.predict(X)

    def run_BN_model(self):
        self.initialized_data()
        self.create_network(self.nodes)
        self.value_meanings.update({
            'HS':{'0':'0:NotHedged','1':'1:Hedged'}
        })
        self.fit_network(self.bn_data)
        if self.portfolio_hedged:
            self.update_network_hedge()
        #self.get_regression_data()