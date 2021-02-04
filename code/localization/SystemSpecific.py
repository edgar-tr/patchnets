
def get_settings_dictionary():
    dict = {}

    dict["default_specs_file"] = "specs.json"
    dict["root_folder"] = "some_folder_for_experiments"
  
    return dict

def system_specific_cleanup():
    pass

def system_specific_session_config():
    dict = {}
    return dict