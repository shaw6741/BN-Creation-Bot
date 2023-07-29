import numpy as np
import pandas as pd
import copy
import datetime
import random

from fredapi import Fred
import yfinance as yf

import pgmpy.models as pm
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import MaximumLikelihoodEstimator, BayesianEstimator, BicScore, K2Score
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination
from pgmpy.metrics import log_likelihood_score

import networkx as nx
import arviz as az
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

yf.pdr_override()


class data_utilities:
    def get_busines_day(self, BusDate):
        new_date = copy.deepcopy(BusDate)
        if datetime.date.weekday(new_date) == 5:
            new_date = new_date - datetime.timedelta(days = 1)
        elif datetime.date.weekday(new_date) == 6:
            new_date = new_date - datetime.timedelta(days = 2)
        return new_date

    def get_start_end_dates(self, time_horizon=5):
        end_date = datetime.date.today() - datetime.timedelta(days = 1)
        end_date = self.get_busines_day(end_date)

        start_date = end_date - datetime.timedelta(days = 365 * time_horizon)
        start_date = self.get_busines_day(start_date)
        return start_date, end_date

    def get_rolling_returns(self, df, periods=1, roll_window=5):
        ret_df = copy.deepcopy(df).pct_change(periods).dropna()
        rolling_ret_df = copy.deepcopy(ret_df).rolling(window=roll_window).sum().dropna()
        return ret_df, rolling_ret_df

    def get_mc_correction(self, df, col_name='MC'):
        """0: not mc, 1: mc of 5%, 2 mc of 10% and 3 mc of 20%"""
        l0 = df <= 0.0
        l1 = df <= -0.05
        l2 = df <= -0.1
        l3 = df <= -0.2

        l1 = l1.astype(int)
        l2 = l2.astype(int)
        l3 = l3.astype(int)

        mc = l0 + l1 + l2 + l3
        mc = pd.DataFrame(mc)
        mc.columns = [col_name]
        value_meanings = {
            'MC':{'0':'0:NoMC','1':'1:MCof5%','2':'2:MCof10%','3':'3:MCof20%'}
        }
        return mc, value_meanings

    def get_portfolio_loss(self, df, port_loss, col_name, side):
        """0: not loss, 1: loss"""
        ret = copy.deepcopy(df)

        if side:
            corrections = ret <= port_loss
        else:
            corrections = -1 * ret <= port_loss
        port_df = pd.DataFrame(corrections)
        port_df.columns = [col_name]
        port_df[col_name] = port_df[col_name].astype(int)
        value_meanings = {
            col_name:{'0':f'0:loss<={abs(port_loss)}', 
                      '1':f'1:loss>{abs(port_loss)}'}
        }
        return port_df, value_meanings


class get_data(data_utilities):
    def __init__(self):
        self.fred = Fred(api_key='d68ec16bf05ee54e7c928ff6d7e144e3')
        self.market_ticker_dict = {'SP': '^GSPC', 'NASDAQ': '^IXIC', 'DOW': '^DJI', 'Russell': '^RUT', 'VIX': '^VIX',
                                   'CRUDE': 'CL=F', 'OIL': 'CL=F', 'OSD': 'CL=F', 'GOLD': 'GC=F', 'SILVER': 'SI=F', 'BOND': '^TNX', 'NOTE': '^TNX', 'FED': 'ZQ=F',
                                   'Financial': 'XLF', 'Materials': 'XLB', 'Communications': 'XLC', 'Energy': 'XLE',
                                   'Industrial': 'XLI', 'Technology': 'XLK', 'Consumer_Staples': 'XLP', 'Real_Estate': 'XLRE',
                                   'Utilities': 'XLU', 'Healthcare': 'XLV', 'Consumer_Discretionary': 'XLY',
                                   'Growth': 'VUG', 'Value': 'VTV', 'Small_Cap': 'VB', 'Mid_Cap': 'VO', 'Large_Cap': 'VV', }
        self.economical_ticker_dict = {'CPI': 'CPIAUCSL', 'INF': 'CPIAUCSL', 'IN': 'CPIAUCSL', 
                                       'FED': 'DFF', 'FR': 'DFF', 'FRI': 'DFF',
                                       'EPU': 'USEPUINDXD',
                                       'Unemployment': 'UNRATE', 'UNP': 'UNRATE'}

    def get_historical_data(self, ticker='SP', start_date='1970-01-01', end_date='2023-06-01', real_ticker=None):
        data = []
        if real_ticker == None:
            if ticker in self.market_ticker_dict.keys():
                data = self.get_yahoo(self.market_ticker_dict[ticker], start_date, end_date)
            elif ticker in self.economical_ticker_dict.keys():
                data = self.get_fred(self.economical_ticker_dict[ticker], start_date, end_date)
            else:
                try:
                    data = self.get_yahoo(ticker, start_date, end_date)
                except:
                    print('Error: ticker not found ' + ticker)
        else:
            try:
                data = self.get_yahoo(real_ticker, start_date, end_date)
            except:
                print('not yahoo')
            if len(data) == 0:
                try:
                    data = self.get_fred(real_ticker, start_date, end_date)
                except:
                    print('not fred')
        return data

    def get_yahoo(self, ticker, start_date, end_date):
        all_df = yf.download(ticker, start= start_date, end= end_date).dropna()
        close_df = all_df['Close']
        close_df.index = pd.to_datetime(close_df.index)
        return close_df

    def get_fred(self, ticker, start_date, end_date):
        fred_df = self.fred.get_series(ticker, observation_start=start_date, observation_end=end_date)
        fred_df.index = pd.to_datetime(fred_df.index)
        return fred_df

    def create_mixed_index(self, stockA_df, stockB_df, p_a=0.5, p_b=0.5, inv=10000.0):
        df = pd.concat([stockA_df, stockB_df], axis=1).dropna()
        df.columns = ['Stock_A', 'Stock_B']
        price_a = float(df.Stock_A.iloc[0])
        price_b = float(df.Stock_B.iloc[0])
        size_a = (inv * p_a) / price_a
        size_b = (inv * p_b) / price_b
        df['Close'] = (df['Stock_A'] * size_a + df['Stock_B'] * size_b) / 100.0
        return df['Close']

    def get_df_portfolio(self, client_portfolio, start, end):
        if client_portfolio in self.market_ticker_dict.keys():
            port_df = self.get_historical_data(client_portfolio, start, end)

        elif client_portfolio != 'Core' and 'Core' in client_portfolio:
            etf_g = self.get_historical_data('Growth', start, end)
            etf_v = self.get_historical_data('Value', start, end)
            etf_a = self.create_mixed_index(etf_g, etf_v)

            if client_portfolio == 'Small_Core':
                index_b = 'Small_Cap'
            elif client_portfolio == 'Mid_Core':
                index_b = 'Mid_Cap'
            elif client_portfolio == 'Large_Core':
                index_b = 'Large_Cap'
            else:
                print('Error, portfolio not recognized: ' + client_portfolio)

            etf_b = self.get_historical_data(index_b, start, end)
            port_df = self.create_mixed_index(etf_a, etf_b)

        else:
            if client_portfolio == 'Core':
                index_a = 'Growth'
                index_b = 'Value'
            elif client_portfolio == 'Small_Growth':
                index_a = 'Small_Cap'
                index_b = 'Growth'
            elif client_portfolio == 'Mid_Growth':
                index_a = 'Mid_Cap'
                index_b = 'Growth'
            elif client_portfolio == 'Large_Growth':
                index_a = 'Large_Cap'
                index_b = 'Growth'
            elif client_portfolio == 'Small_Value':
                index_a = 'Small_Cap'
                index_b = 'Value'
            elif client_portfolio == 'Mid_Value':
                index_a = 'Mid_Cap'
                index_b = 'Value'
            elif client_portfolio == 'Large_Value':
                index_a = 'Large_Cap'
                index_b = 'Value'
            else:
                print('Error, portfolio not recognized: ' + client_portfolio)

            etf_a = self.get_historical_data(index_a, start, end)
            etf_b = self.get_historical_data(index_b, start, end)
            port_df = self.create_mixed_index(etf_a, etf_b)

        return port_df


class call_arviz_lib:
    def get_summary(self, data, round_to=2):
        return az.summary(data, round_to=round_to)

    def get_plot_trace(self, data, compact=False):
        az.plot_trace(data, compact=compact, figsize=(8, 6))

    def get_autocorrelation(self, data, max_lag=30, combined=True):
        az.plot_autocorr(data, max_lag=max_lag, combined=combined)

    def get_plot_ess(self, data, kind):
        az.plot_ess(data, kind=kind)

    def get_plot_forest(self, data, kind='ridgeplot', hdi_prob=0.95, r_hat=True, ess=True):
        az.plot_forest(data, hdi_prob=hdi_prob, r_hat=r_hat, ess=ess, kind='ridgeplot')

    def get_ess(self, data, var_names):
        return az.ess(data, var_names=var_names)

    def get_rhat(self, data, var_names):
        return az.rhat(data, var_names=var_names)

    def get_plot_mcse(self, data, extra_methods=True):
        az.plot_mcse(data, extra_methods=extra_methods)

    def get_posterior(self, data):
        fig, ax = plt.subplots(figsize=(8, 6), dpi=200)
       

        az.plot_posterior(data)
        # ax.set_xlabel("X-axis label")
        # ax.set_ylabel("Y-axis label")
        ax.set_title("Posterior Distribution")

        plt.show()

    def get_dist(self, data):
        az.plot_dist(data)


class definition_BayesianNetwork:
    def create_network(self, nodes):
        self.model = BayesianNetwork(nodes)

    def create_simple_cpd(self, node_name, num_cards, vals):
        _cpd = TabularCPD(variable=node_name, variable_card=num_cards, values=vals)
        return _cpd

    def create_output_cpd(self, node_name, num_cards, vals, evidence, evid_cards):
        _cpd = TabularCPD(variable=node_name, variable_card=num_cards, values=vals, evidence=evidence,
                          evidence_card=evid_cards)
        return _cpd

    def fit_network(self, df):
        self.model.fit(df)

    def fit_update_network(self, df):
        self.model.fit_update(df)

    def get_predict(self, predict_data):
        y_pred = self.model.predict(predict_data)
        return y_pred

    def get_predict_probability(self, predict_data):
        y_prob = self.model.predict_probability(predict_data)
        return y_prob

    def model_add_cpds(self, cpd):
        self.model.add_cpds(cpd)

    def model_remove_cpds(self, cpd):
        self.model.remove_cpds(cpd)

    def inference_object(self, var_name, evidence):
        infer = VariableElimination(self.model)
        posterior_prob = infer.query(variables=[var_name], evidence=evidence)
        return posterior_prob

    def check_model(self):
        #hist
        return self.model.check_model()

    def model_simulate(self, samples):
        return self.model.simulate(n_samples=samples)

    def get_nodes(self):
        return self.model.nodes()

    def get_edges(self):
        return self.model.edges()

    def get_cpds(self):
        return self.model.get_cpds()

    def get_cardinality(self):
        return self.model.get_cardinality()

    def get_mle(self, data):
        mle = MaximumLikelihoodEstimator(self.model, data)
        return mle

    def get_bic_est(self, data):
        self.bic = BicScore(data)
        return self.bic.score(self.model)

    def get_k2_est(self, data):
        self.k2 = K2Score(data)
        return self.k2.score(self.model)

    def get_log_l_score(self, data):
        return log_likelihood_score(self.model, data)

    def to_mm_model(self):
        self.mm_model = self.model.to_markov_model()

    def plot_daft(self):
        return self.model.to_daft().render()
    
    def plot_networkx(self):
        nx_graph = nx.DiGraph(self.model.edges())
        for layer, nodes in enumerate(nx.topological_generations(nx_graph)):
            for node in nodes:
                nx_graph.nodes[node]["layer"] = layer
        pos = nx.multipartite_layout(nx_graph, subset_key="layer")
        fig, ax = plt.subplots()
        nx.draw_networkx(nx_graph, pos=pos, ax=ax)
        return fig


def get_hist_data_from_BN(raw_data):
    csv_path = './utils/hist_mc.csv'
    train_data = pd.read_csv(csv_path)
    bn_class = definition_BayesianNetwork()
    bn_class.create_network(nodes=[['DW', 'MC']])
    bn_class.fit_network(train_data)

    test_data = pd.DataFrame(list(raw_data.values), columns=['MC'])
    sim_data = bn_class.model_simulate(len(test_data))
    sim_array = np.array(sim_data['DW'])
    war_value_meanings = {
        'DOW':{'2':'2:20DaysPostwar',
              '1':'1:WarStart',
              '0':'0:BeforeWar'},
        'TW':{'2':'2:20DaysPostwar',
              '1':'1:WarStart',
              '0':'0:BeforeWar'},
        'DW':{'2':'2:20DaysPostwar',
              '1':'1:WarStart',
              '0':'0:BeforeWar'}
    }
    return sim_array, war_value_meanings
