import json

class NSGAConfig:
    def __init__(self, config_file):
        self.config_file = config_file
        self.config_data = self.load_config()

    def load_config(self):
        with open(self.config_file, 'r') as file:
            config_data = json.load(file)
        return config_data
    
    def get_parameters(self,dataset_name):
        population_size = self.config_data[dataset_name]['population_size']
        Q = self.config_data[dataset_name]['Q']
        LP = self.config_data[dataset_name]['LP']
        igd_threshold = self.config_data[dataset_name]['igd_threshold']
        hv_threshold = self.config_data[dataset_name]['hv_threshold']
        no_improvement_limit = self.config_data[dataset_name]['no_improvement_limit']
        maxFEs = self.config_data[dataset_name]['maxFEs']
        return population_size, Q, LP, igd_threshold, hv_threshold, no_improvement_limit, maxFEs