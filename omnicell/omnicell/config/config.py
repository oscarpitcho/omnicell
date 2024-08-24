

#TODO: - Configs just for trainings, configs for evaluations, 

class Config:
    def __init__(self, config_file):
        self.config = configparser.ConfigParser()

        self.task_config = None 
        self.model_config = None
        self.data_config = None
        self.config.read(config_file)



    #Several of these functions for the different places where one might instantiate a config object
    def from_task_config(task_config):
        pass





    
    def get_data_path(self):
        return self.config['data']['path']
    
    def get_eval_targets(self):
        return self.config['datasplits']['eval_targets']
    
    def get_model_name(self):
        return self.config['model']['name']
    
    def get_task_name(self):
        if self.task_config is None:
            raise ValueError("No task config found in the config file")
        
        return self.config['task']['name']
    
    def get_heldout_cells(self):
        return self.config['datasplits']['heldout_cells']
    
    def_