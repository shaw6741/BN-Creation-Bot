from baynet.bayesian_model import model_run
import json

class read_json():
    def __init__(self):
        # self.path = '/Users/zy/Desktop/capstone 2/conversation.json'  # Yao
        #self.path = 'D:/Documents/UChicago/capstone/engine/conversation.json'  # Oscar
        self.path = 'E:/px/UChi/Courses/Capstone/code/engine/conversation.json' # Xiao
        self.data = None

        # Inputs from conversation
        self.mkt_ticker = 'SP'
        self.client_portfolio = 'Mid_Core'
        self.input_nodes = []

        # Our default inputs for BN and results
        self.oil_jump = 0.1
        self.inflation_jump = 0.002
        self.crash_ = -0.1
        self.portfolio_loss = -0.05
        self.simulate_samples_number = 10000
        self.historical_time_horizon = 2
        self.diff_periods = 1
        self.returns_roll_window = 5

    def read_(self):
        with open(self.path, 'r') as file:
            self.data = json.load(file)

        self.filter_empty_keys()

    def filter_empty_keys(self):
        keys_to_check = ['control', 'mkt_cap', 'mitigators', 'triggers', 'events', 'edges', 'consequences', 'style', 'sectors']
        for key in keys_to_check:
            value = self.data.get(key)
            if value is not None:
                setattr(self, key, value)


class Engine(read_json):
    def __init__(self):
        super().__init__()
        self.read_()

    def read_(self):
        super().read_()
        self.mkt_ticker = (self.data.get('mkt_cap'))
        self.client_portfolio = (self.data.get('style'))
        self.input_nodes = (self.data.get('edges'))
        self.triggers_dict = (self.data.get('triggers'))
        self.controls_dict = (self.data.get('controls'))
        self.events_dict = (self.data.get('events'))
        self.mitigators_dict = (self.data.get('mitigators'))
        self.consequences_dict = (self.data.get('consequences'))

    def start(self):
        self.BN_model = model_run(self.mkt_ticker, self.client_portfolio, self.input_nodes, self.crash_, self.portfolio_loss,
                                  self.oil_jump, self.inflation_jump, self.diff_periods,  self.historical_time_horizon, self.returns_roll_window,
                                  self.triggers_dict, self.controls_dict, self.events_dict, self.mitigators_dict, self.consequences_dict)
        self.BN_model.run_BN_model()

