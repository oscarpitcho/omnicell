
import json

class Catalogue: 
    def __init__(self, path_to_catalogue):
        self.catalogue = json.load(open(path_to_catalogue))


    

    def get_dataset_names(self):
        return [x['name'] for x in self.catalogue['datasets']]

    "Might be useful for some script down the line"
    def register_new_dataset(self, name, path):
        pass