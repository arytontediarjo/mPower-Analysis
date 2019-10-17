import pandas as pd
import numpy as np
import json 
import os
import ast
from synapseclient import Entity, Project, Folder, File, Link, Activity


"""
Query for synapse table entity (can be used for tremors and walking V1 and V2)
parameter:  syn: synapse object, 
            healthcodes: list of objects, 
            synId: table that you want to query from
returns: a dataframe of healthCodes and their respective filepaths 
"""
def get_synapse_table(syn, healthcodes, synId):
    ### get healthcode subset from case-matched tsv, or any other subset of healthcodes
    healthcode_subset = "({})".format([i for i in healthcodes]).replace("[", "").replace("]", "")   
    ### query from synapse and download to synapsecache ### 
    query = syn.tableQuery("select * from {} WHERE healthCode in {} LIMIT 10".format(synId, healthcode_subset))
    data = query.asDataFrame()
    json_list = [_ for _ in data.columns if "json" in _]
    file_map = syn.downloadTableColumns(query, json_list)
    ### Loop through the dictionary ### 
    dict_ = {}
    dict_["file_handle_id"] = []
    dict_["file_path"] = []
    for k, v in file_map.items():
        dict_["file_handle_id"].append(k)
        dict_["file_path"].append(v)
    filepath_data = pd.DataFrame(dict_)
    data = data[["recordId","phoneInfo", "createdOn", "healthCode"] + json_list]
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
        except:
            return "#ERROR"
        data = clean_accelerometer_data(data)
        return data[["td","x", "y", "z", "AA"]]
        
    ## userAcceleration from mpowerV1
    elif "userAcceleration" in data.columns:
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
    return data

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
def get_healthcodes(syn, synId, is_filtered):
    ## get demographic information
    if is_filtered:
        filtered_entity = syn.get("syn8381056")
        healthcode_list = list(pd.read_csv(filtered_entity["path"], sep = "\t")["healthCode"])
        return healthcode_list
    else:
        healthcode_list = list(syn.tableQuery("select distinct(healthCode) as healthCode from {}".format(synId))
                                   .asDataFrame()["healthCode"])
        return healthcode_list
    
    
""" 
function to retrieve sensors 
"""
def get_sensor_types(filepath):
    print(filepath)
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
function to store to synapse with provenance 
"""
def store_to_synapse(syn, 
                     filename, 
                     data,
                     parentId,
                     **activities):
    
    ## name of the output file ##
    file_path = filename
    
    ## set activity entity for provenance ##
    activity = Activity(
        name     = activities.get("name"),
        executed = activities.get("script_id"),
        used     = activities.get("source_id"))
    
    ## condition for storing scripts ##
    if ("py" in file_path.split(".")) or ("R" in file_path.split(".")):
        new_file = File(path = file_path, parentId = parentId)
        new_file = syn.store(new_file, activity = activity)
    
    ## condition for storing csv data *will be improved with other formats* ##
    else:
        data = data.to_csv(file_path)
        new_file = File(path = file_path, parentId = parentId)
        new_file = syn.store(new_file, activity = activity)
        os.remove(file_path)

## function to retrieve synId of a scripts ##
def get_script_id(syn, filename, parentId):
    #   get list of files
    file_list = list(syn.getChildren(parent = parentId, includeTypes = ['file']))
    #   iterate through children
    for dict_ in file_list:
        if dict_["name"] == filename:
            return dict_["id"]
    #   file not available
    return np.NaN 

def map_to_json(params):
    if isinstance(params, dict):
        return params
    else:
        return np.NaN
    
def normalize_feature(data, feature):
    normalized_data = data[feature].map(map_to_json) \
                                .apply(pd.Series) \
                                .fillna("#ERROR").add_prefix('{}.'.format(feature))
    data = pd.concat([data, normalized_data], axis = 1).drop(feature, axis = 1)
    return data


def generate_provenance(syn, filename, 
                        data, pyfile, 
                        synId, **parentId):
    ## store data and script ##
        path_to_script = os.path.join(os.getcwd(), pyfile)
        output_filename = os.path.join(os.getcwd(), filename)
        store_script = store_to_synapse(syn  = syn, filename = path_to_script,
                                    data = np.NaN, parentId = parentId.get("script_id"))
        store_data = store_to_synapse(syn  = syn, filename  = output_filename,
                                  data = data, parentId = parentId.get("data_id"),
                                  source_id = synId, name = "feature preprocessing",
                                  script_id = get_script_id(syn, __file__, parentId.get("script_id")))
    
    
    
    

    