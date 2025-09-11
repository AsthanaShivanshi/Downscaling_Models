def load_config(main_path:str, private_path:str)->dict:
    import yaml
    from collections.abc import Mapping

    def filepaths_merge(source:dict, overrides:dict)-> dict:
        for key,value in overrides.items():
            if isinstance(value, Mapping) and key in source:
                source[key]=filepaths_merge(source.get(key,{}),value)
            else:
                source[key]=value
        return source

    with open(main_path,"r") as f:
        config=yaml.safe_load(f)
    with open(private_path,"r") as f:
        paths=yaml.safe_load(f)

    # If paths has a top-level 'data', merge that into config['data']
    if "data" in paths:
        config["data"] = filepaths_merge(config.get("data", {}), paths["data"])
    else:
        config["data"] = filepaths_merge(config.get("data", {}), paths)

    return config