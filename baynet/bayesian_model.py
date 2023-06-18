import numpy as np
import pandas as pd
import copy
import datetime
import random
from utils.utils import get_data, definition_BayesianNetwork


class model_run(definition_BayesianNetwork):
    def __init__(self, mkt_ticker, client_portfolio, nodes, crash, portfolio_loss, oil_jump, inf_jump, diff_periods, time_horizon, roll_win,
                 triggers_dict, controls_dict, events_dict, mitigators_dict, consequences_dict):
        self.mkt_ticker = mkt_ticker
        self.client_portfolio = client_portfolio
        self.nodes = nodes
        self.crash_ = crash
        self.portfolio_loss = portfolio_loss
        self.oil_jump = oil_jump
        self.inf_jump = inf_jump
        self.diff_periods = diff_periods
        self.time_horizon = time_horizon
        self.roll_win = roll_win

        if triggers_dict:
            self.triggers_list = list(triggers_dict.keys())
        if controls_dict:
            self.controls_list = list(controls_dict.keys())
        if events_dict:
            self.events_list = list(events_dict.keys())
        if mitigators_dict:
            self.mitigators_list = list(mitigators_dict.keys())
        if consequences_dict:
            self.consequences_list = list(consequences_dict.keys())

        self.indicators_node_list = []
        self.missing_node_list = []
        self.not_indicator_node_list = []
        self.eco_data = pd.DataFrame()

        self.oil_syn = ['CRUDE', 'OIL', 'OSD']
        self.inf_syn = ['INF', 'CPI']
        self.fed_syn = ['FED']
        self.unemployment_syn = ['UNP']
        self.buy_straddle_syn = ['BS']
        self.short_sell_syn = ['SF', 'SS']

        self.data_pull = get_data()
        self.start, self.end = self.data_pull.get_start_end_dates(self.time_horizon)

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

    def construct_neg_ret(self, ticker, perc_loss, col_name):
        df_close = self.data_pull.get_df_portfolio(ticker, self.start, self.end)
        df_ret, df_roll_ret = self.data_pull.get_rolling_returns(df_close, self.diff_periods, self.roll_win)
        df_correction = self.data_pull.get_mc_correction(df_roll_ret, perc_loss, col_name)
        return df_correction

    def construct_mkt_crash(self):
        if self.events_list:
            self.mc_df = self.construct_neg_ret(self.mkt_ticker, self.crash_, self.events_list[0])
            self.not_indicator_node_list.append(self.events_list[0])
        else:
            self.mc_df = self.construct_neg_ret(self.mkt_ticker, self.crash_, 'MC')
            self.not_indicator_node_list.append('MC')

    def construct_portfolio_loss(self):
        if self.consequences_list:
            self.port_c_df = self.construct_neg_ret(self.client_portfolio, self.portfolio_loss, self.consequences_list[0])
            self.not_indicator_node_list.append(self.consequences_list[0])
        else:
            self.port_c_df = self.construct_neg_ret(self.client_portfolio, self.portfolio_loss, 'PIL')
            self.not_indicator_node_list.append('PIL')

    def merge_df(self):
        self.merged_data = pd.merge(self.mc_df, self.eco_data, how='left', left_index=True, right_index=True)
        self.merged_data.fillna(method='ffill', inplace=True)
        self.merged_data.dropna(inplace=True)

        self.merged_data = pd.merge(self.merged_data, self.port_c_df, how='left', left_index=True, right_index=True)
        self.merged_data.fillna(method='ffill', inplace=True)
        self.merged_data.dropna(inplace=True)

    def generate_bin_data(self):
        self.bn_data = copy.deepcopy(self.merged_data)

        for ind in self.indicators_node_list:
            if ind in self.oil_syn:
                self.bn_data[ind] = (self.bn_data[ind] >= self.oil_jump) | (self.bn_data[ind] <= -self.oil_jump)
            elif ind in self.inf_syn:
                self.bn_data[ind] = self.bn_data[ind] >= self.inf_jump
            else:
                self.bn_data[ind] = self.bn_data[ind] >= 0
            self.bn_data[ind] = self.bn_data[ind].astype(int)

    def get_rnd(self, n, pct_0=60):
        random_array = np.random.choice([0, 1], size=n, p=[pct_0 / 100, (100 - pct_0) / 100])
        return random_array

    def generate_missing_data(self):
        # get some random date for other parameters
        N = len(self.bn_data)
        for col_name in self.missing_node_list:
            if col_name in self.buy_straddle_syn:
                bs_data = self.data_pull.get_historical_data('VIX', self.start, self.end)
                _, bs_ret_roll = self.data_pull.get_rolling_returns(bs_data, self.diff_periods, self.roll_win)
                self.merged_data[col_name] = bs_ret_roll
                self.bn_data[col_name] = bs_ret_roll >= abs(self.crash_)
                self.bn_data[col_name] = self.bn_data[col_name].astype(int)
            elif col_name in self.short_sell_syn:
                ss_data = self.data_pull.get_historical_data('SP', self.start, self.end)
                _, ss_ret_roll = self.data_pull.get_rolling_returns(ss_data, self.diff_periods, self.roll_win)
                self.merged_data[col_name] = ss_ret_roll
                self.bn_data[col_name] = ss_ret_roll <= 0
                self.bn_data[col_name] = self.bn_data[col_name].astype(int)
            else:
                rand_num = random.randint(50, 99)
                temp_array = self.get_rnd(N, rand_num)
                self.bn_data[col_name] = temp_array

    def initialized_data(self):
        self.construct_mkt_crash()
        self.construct_portfolio_loss()
        self.get_indicators_node_list()

        for ind in self.indicators_node_list:
            ind_data = self.data_pull.get_historical_data(ind, self.start, self.end)
            ind_ret, int_roll_ret = self.data_pull.get_rolling_returns(ind_data, self.diff_periods, self.roll_win)
            if ind in self.inf_syn:
                self.eco_data = pd.concat([self.eco_data, ind_ret], axis=1)
            else:
                self.eco_data = pd.concat([self.eco_data, int_roll_ret], axis=1)
        self.eco_data.columns = self.indicators_node_list

        self.merge_df()
        self.generate_bin_data()
        self.generate_missing_data()

    def run_BN_model(self):
        self.initialized_data()
        self.create_network(self.nodes)
        self.fit_network(self.bn_data)