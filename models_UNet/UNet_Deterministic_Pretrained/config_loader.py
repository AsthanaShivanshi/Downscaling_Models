#This script is being used for merging the yaml configuration file with the dataset paths

import yaml
from collections.abc import Mapping

def filepaths_merge(source:dict, overrides:dict)-> dict:
    for key,value in overrides.items():
        if isinstance(value, Mapping) and key in source:
            source[key]=filepaths_merge(source.get(key,{}),value)
        else:
            source[key]=value #Updating the dictionary merge with key value pairs


    return source

#load_config is for merging the dataset paths with the YAML config file
def load_config(main_path:str, private_path:str)->dict:
    with open(main_path,"r") as f:
        config=yaml.safe_load(f)
    with open(private_path,"r") as f:
        paths=yaml.safe_load(f)

    config["data"]= filepaths_merge(config.get("data",{}),paths)

    return config