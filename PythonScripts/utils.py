import sys
import json 
import os
import ast
import pandas as pd
import numpy as np
import synapseclient as sc
from synapseclient import Entity, Project, Folder, File, Link, Activity

## instantiate syn global variable ##
syn = sc.login()

"""
Query for synapse table entity (can be used for tremors and walking V1 and V2)
parameter:  syn: synapse object, 
            healthcodes: list of objects
returns: a dataframe of healthCodes and their respective filepaths 
"""
def get_walking_synapse_table(healthcodes, table_id, version):
    
    ## check syn object ##
    if ("syn" not in globals()):
        syn = sc.login()
    else:
        syn = globals()["syn"]
    
    ### get healthcode subset from case-matched tsv, or any other subset of healthcodes
    healthcode_subset = "({})".format([i for i in healthcodes]).replace("[", "").replace("]", "")   
    ### query from synapse and download to synapsecache ###     
    if version == "V1":
        print("Querying V1 Data")
        query = syn.tableQuery("select * from {} WHERE healthCode in {}".format(table_id, healthcode_subset))
        data = query.asDataFrame()
        json_list = [_ for _ in data.columns if ("deviceMotion" in _)]
    elif version == "V2":
        print("Querying V2 Data")
        query = syn.tableQuery("select * from {} WHERE healthCode in {}".format(table_id, healthcode_subset))
        data = query.asDataFrame()
        json_list = [_ for _ in data.columns if ("json" in _)]
    else:
        print("Querying Passive Data")
        query = syn.tableQuery("select * from {} WHERE healthCode in {}".format(table_id, healthcode_subset))
        data = query.asDataFrame()
        json_list = [_ for _ in data.columns if ("json" in _)]   
        
    ### Download tmp into ordered dictionary ###
    file_map = syn.downloadTableColumns(query, json_list)
    ### Loop through the dictionary ### 
    dict_ = {}
    dict_["file_handle_id"] = []
    dict_["file_path"] = []
    for k, v in file_map.items():
        dict_["file_handle_id"].append(k)
        dict_["file_path"].append(v)
    filepath_data = pd.DataFrame(dict_)
    data = data[["recordId", "healthCode", "appVersion", "phoneInfo", "createdOn"] + json_list]
    filepath_data["file_handle_id"] = filepath_data["file_handle_id"].astype(float)
    
    ### Join the filehandles with each acceleration files ###
    for feat in json_list:
        data[feat] = data[feat].astype(float)
        data = pd.merge(data, filepath_data, 
                        left_on = feat, 
                        right_on = "file_handle_id", 
                        how = "left")
        data = data.rename(columns = {feat: "{}_path_id".format(feat), 
                            "file_path": "{}_pathfile".format(feat)}).drop(["file_handle_id"], axis = 1)
    data["createdOn"] = pd.to_datetime(data["createdOn"], unit = "ms")
    
    ## Empty Filepaths ##
    data = data.fillna("#ERROR") 
    return data


"""
Function to produce accelerometer data
parameter: filepath: filepaths of given data
returns:    a tidied version of the dataframe that contains a time-index dataframe, time differences,
            (x, y, z, AA) acceleration
"""
def get_acceleration_ts(filepath): 
    ## if empty filepaths return it back
    if filepath == "#ERROR":
        return filepath
    
    ## open filepath
    data = open_filepath(filepath)
    
    ## return accelerometer data back if empty ##
    if data.shape[0] == 0 or data.empty: 
        return "#ERROR"
    
    ## get data from mpowerV2, annotated by the availability of sensorType
    if ("sensorType" in data.columns):
        try:
            data = data[data["sensorType"] == "userAcceleration"]
            data = clean_accelerometer_data(data)
        except:
            return "#ERROR"
        return data[["td","x", "y", "z", "AA"]]
        
    ## userAcceleration from mpowerV1
    elif ("userAcceleration" in data.columns):
        data = data[["timestamp", "userAcceleration"]]
        data["x"] = data["userAcceleration"].apply(lambda x: x["x"])
        data["y"] = data["userAcceleration"].apply(lambda x: x["y"])
        data["z"] = data["userAcceleration"].apply(lambda x: x["z"])
        data = data.drop(["userAcceleration"], axis = 1)
        data = clean_accelerometer_data(data)
        return data[["td","x", "y", "z", "AA"]]
    
"""
Generalized function to clean accelerometer data 
format: time index, time difference from point zero, (x, y, z, AA) vector coordinates
"""
def clean_accelerometer_data(data):
    data = data.dropna(subset = ["x", "y", "z"])
    date_series = pd.to_datetime(data["timestamp"], unit = "s")
    data["td"] = date_series - date_series.iloc[0]
    data["td"] = data["td"].apply(lambda x: x.total_seconds())
    data["time"] = data["td"]
    data = data.set_index("time")
    data.index = pd.to_datetime(data.index, unit = "s")
    data["AA"] = np.sqrt(data["x"]**2 + data["y"]**2 + data["z"]**2)
    data = data.sort_index()
    
    ## check if datetime index is sorted
    if all(data.index[:-1] <= data.index[1:]):
        return data 
    else:
        sys.exit('Time Series File is not Sorted')

"""
General Function to open a filepath 
and returns a dataframe 
"""
def open_filepath(filepath):
    with open(filepath) as f:
        json_data = f.read()
        data = pd.DataFrame(json.loads(json_data))
    return data


"""
General Function get filtered healthcodes or all the healthcodes
parameter:  syn: syn object,
            filter: file of the filtered healthcodes,
            synId: table that user want to query from,
            is_filtered: boolean values of whether user wants to have filtered healthcode queries
returns list of healthcodes
"""
def get_healthcodes(table_id, is_filtered):
    ## check syn object ##
    if ("syn" not in globals()):
        syn = sc.login()
    else:
        syn = globals()["syn"]
    ## get demographic information
    if is_filtered:
        filtered_entity = syn.get("syn8381056")
        healthcode_list = list(pd.read_csv(filtered_entity["path"], sep = "\t")["healthCode"])
        return healthcode_list
    else:
        healthcode_list = list(syn.tableQuery("select distinct(healthCode) as healthCode from {}".format(table_id))
                                   .asDataFrame()["healthCode"])
        return healthcode_list
    
    
""" 
function to retrieve sensors 
"""
def get_sensor_types(filepath):
    if filepath == "#ERROR":
        return filepath
    data = open_filepath(filepath)
    if "sensorType" in data.columns:
        return data["sensorType"].dropna().unique()
    else:
        return "#ERROR"

"""
function to retrieve unit of measurements 
"""
def get_units(filepath):
    if filepath == "#ERROR":
        return filepath
    data = open_filepath(filepath)
    if "unit" in data.columns:
        return data["unit"].dropna().unique()
    else:
        return "#ERROR"
    
"""
function for retrieving specs (if available)
"""
def get_sensor_specs(filepath):
    if filepath == "#ERROR":
        return filepath
    data = open_filepath(filepath)
    if "sensor" in data.columns:
        return data["sensor"].iloc[0]
    else:
        return "#ERROR"
    
    

"""
 Function to retrieve synId of a scripts 
"""
def get_script_id(path_to_script, script_parent_Id):
    
    ## check syn object ##
    if ("syn" not in globals()):
        syn = sc.login()
    else:
        syn = globals()["syn"]
    
    ##  save file to synapse first ##
    new_script = File(path = path_to_script, parentId = script_parent_Id)
    syn.store(new_script)   
    
    ##  iterate through children to find the synId of saved script ##
    for dict_ in list(syn.getChildren(parent = script_parent_Id, includeTypes = ['file'])):
        if dict_["name"] == path_to_script.split("/")[-1]:
            return dict_["id"]
    
    ##  file not available ##
    raise Exception("Check script name in %s" %script_parent_Id)



"""
Function to save to synapse 
params: data
"""
def save_to_synapse(data, used_script,
                    walking_table_id,
                    data_parent_id, 
                    script_parent_id, 
                    output_filename): 

        ## check if syn object is a global variable ##
        if ("syn" not in globals()):
            syn = sc.login()
        else:
            syn = globals()["syn"]
        
        ## set path to script and output filename ##
        path_to_script = os.path.join(os.getcwd(), used_script)
        path_to_output_filename = os.path.join(os.getcwd(), output_filename)
        
        ## save the script to synapse ##
        data     = data.to_csv(path_to_output_filename)
        new_file = File(path = path_to_output_filename, parentId = data_parent_id)
        act      = Activity(used  = walking_table_id,
                            executed = get_script_id(used_script, script_parent_id))
        new_file = syn.store(new_file, activity = act)           
        os.remove(output_filename)

"""
Function to check json 
"""
def map_to_json(params):
    if isinstance(params, dict):
        return params
    else:
        return "#ERROR"


"""
Function to normalize pdkit dictionaries to columns
Fill none as 0 meaning that the signal is too weak to be detected
"""    
def normalize_feature(data, feature):
    normalized_data = data[feature].map(map_to_json) \
                                .apply(pd.Series) \
                                .fillna("#ERROR").add_prefix('{}.'.format(feature))
    data = pd.concat([data, normalized_data], 
                     axis = 1).drop(feature, axis = 1)
    return data


def fix_column_name(data):
    for feature in filter(lambda x: "feature" in x, data.columns): 
        data  = data.rename({feature: "%s"\
                            %(feature.split("features_")[1])}, axis = 1)
    return data

    