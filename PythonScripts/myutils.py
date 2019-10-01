import pandas as pd
import numpy as np
import json 
import os
import ast

"""
Prototype:
Query for synapse table entity (can be used for tremors and walking V1 and V2)
parameter: syn: synapse object, healthcodes: list of objects, synId: table entity that you want
returns: a dataframe of healthCodes and their respective filepaths 
"""
def get_synapse_table(syn, healthcodes, synId):
    ### get healthcode subset from case-matched tsv, or any other subset of healthcodes
    healthcode_subset = "({})".format([i for i in healthcodes]).replace("[", "").replace("]", "")
    
    ### query from synapse and download to synapsecache ### 
    query = syn.tableQuery("select * from {} WHERE healthCode in {}".
                       format(synId, healthcode_subset))
    data = query.asDataFrame()
    json_list = [_ for _ in data.columns if "json" in _]
    data[json_list] = data[json_list].applymap(lambda x: str(x))
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
    
    filepath_data["file_handle_id"] = filepath_data["file_handle_id"].astype(str)
    
    ### Join the filehandles with each acceleration files ###
    for feat in json_list:
        data[feat] = data[feat].astype(str)
        data = pd.merge(data, filepath_data, 
                        left_on = feat, 
                        right_on = "file_handle_id", 
                        how = "left")
        data = data.rename(columns = {feat: "{}_path_id".format(feat), 
                            "file_path": "{}_pathfile".format(feat)}).drop(["file_handle_id"], axis = 1)
    data["createdOn"] = pd.to_datetime(data["createdOn"], unit = "ms")
    data = data.fillna("#ERROR") ## Empty Filepaths
    return data


"""
Takes in both filepaths and gait of any mpower versions
"""
def gait_time_series(filepath): 
    ## if empty filepaths return it back
    if filepath == "#ERROR":
        return filepath
    
    ## open filepath ##
    with open(filepath) as f:
        json_data = f.read()
        data = pd.DataFrame(json.loads(json_data))
    
    ## return accelerometer data back if empty ##
    if data.shape[0] == 0:
        return "#ERROR"
    
    ## mpowerV2 daata
    if "sensorType" in data.columns:
        data = clean_accelerometer_data(data)
        return data[["td","sensorType","x", "y", "z", "AA"]]
        
    ## userAcceleration from mpowerV1
    elif "userAcceleration" in data.columns:
        data = data[["timestamp", "userAcceleration"]]
        data["x"] = data["userAcceleration"].apply(lambda x: x["x"])
        data["y"] = data["userAcceleration"].apply(lambda x: x["y"])
        data["z"] = data["userAcceleration"].apply(lambda x: x["z"])
        data = data.drop(["userAcceleration"], axis = 1)
    ## index time series ##
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