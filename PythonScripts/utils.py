import pandas as pd
import numpy as np
import json 
import os


def getSynapseData(syn,healthcodes):
    
    ### query from synapse and download to synapsecache ### 
    query = syn.tableQuery("select * from {} WHERE healthCode in {}".
                       format("syn7222425", healthcodes))
    file_map = syn.downloadTableColumns(query, ["accel_walking_outbound.json.items", "accel_walking_return.json.items", "accel_walking_rest.json.items", 
                                                "deviceMotion_walking_outbound.json.items", "deviceMotion_walking_return.json.items", "deviceMotion_walking_rest.json.items"])
    
    ### Loop through the dictionary ### 
    data = {}
    data["file_handle_id"] = []
    data["file_path"] = []
    for k, v in file_map.items():
        data["file_handle_id"].append(int(k))
        data["file_path"].append(v)
    filepath_data = pd.DataFrame(data)
    sample_df = query.asDataFrame()
    sub_df = sample_df[["recordId", "createdOn", "healthCode", 
                        "accel_walking_outbound.json.items", 
                        "accel_walking_return.json.items", "accel_walking_rest.json.items",
                        "deviceMotion_walking_outbound.json.items", 
                        "deviceMotion_walking_return.json.items", "deviceMotion_walking_rest.json.items"
                       ]]
    
    
    ### Join the filehandles with each acceleration files ###
    data = pd.merge(sub_df, filepath_data, 
                    left_on = "accel_walking_outbound.json.items", 
                    right_on = "file_handle_id", 
                    how = "left")
    data = data.rename(columns = {"accel_walking_outbound.json.items": "accel_outbound_path_id", 
                          "file_path": "accel_outbound_pathfile" }).drop(["file_handle_id"], axis = 1)
    
    data = pd.merge(data, filepath_data, 
                    left_on = "accel_walking_return.json.items", 
                    right_on = "file_handle_id", 
                    how = "left")
    data = data.rename(columns = {"accel_walking_return.json.items": "accel_return_path_id", 
                          "file_path": "accel_return_pathfile" }).drop(["file_handle_id"], axis = 1)
    
    data = pd.merge(data, filepath_data, 
                    left_on = "accel_walking_rest.json.items", 
                    right_on = "file_handle_id", 
                    how = "left")
    data = data.rename(columns = {"accel_walking_rest.json.items": "accel_rest_path_id", 
                          "file_path": "accel_rest_pathfile" }).drop(["file_handle_id"], axis = 1)
    
    data = pd.merge(data, filepath_data, 
                    left_on = "deviceMotion_walking_outbound.json.items", 
                    right_on = "file_handle_id", 
                    how = "left")
    data = data.rename(columns = {"deviceMotion_walking_outbound.json.items": "deviceMotion_outbound_path_id", 
                          "file_path": "deviceMotion_outbound_pathfile" }).drop(["file_handle_id"], axis = 1)
    
    data = pd.merge(data, filepath_data, 
                    left_on = "deviceMotion_walking_return.json.items", 
                    right_on = "file_handle_id", 
                    how = "left")
    data = data.rename(columns = {"deviceMotion_walking_return.json.items": "deviceMotion_return_path_id", 
                          "file_path": "deviceMotion_return_pathfile" }).drop(["file_handle_id"], axis = 1)
    
    data = pd.merge(data, filepath_data, 
                    left_on = "deviceMotion_walking_rest.json.items", 
                    right_on = "file_handle_id", 
                    how = "left")
    data = data.rename(columns = {"deviceMotion_walking_rest.json.items": "deviceMotion_rest_path_id", 
                          "file_path": "deviceMotion_rest_pathfile" }).drop(["file_handle_id"], axis = 1)
    
    data["createdOn"] = pd.to_datetime(data["createdOn"], unit = "ms")
    data = data.fillna("EMPTY FILEPATHS")
    return data

def processAcceleration(filepath):
    ## open filepath ##
    with open(filepath) as f:
        json_data = f.read()
        df = pd.DataFrame(json.loads(json_data))
        
    ## return accelerometer data back if empty ##
    if df.shape[0] == 0:
        return df
    
    ## extra preprocessing condition for deviceMotion user Acceleration
    elif "userAcceleration" in df.columns:
        df = df[["timestamp", "userAcceleration"]]
        df["x"] = df["userAcceleration"].apply(lambda x: x["x"])
        df["y"] = df["userAcceleration"].apply(lambda x: x["y"])
        df["z"] = df["userAcceleration"].apply(lambda x: x["z"])
        df = df.drop(["userAcceleration"], axis = 1) 
    
    ## index time series ##
    date_series = pd.to_datetime(df["timestamp"], unit = "s")
    df["td"] = date_series - date_series.iloc[0]
    df["td"] = df["td"].apply(lambda x: x.total_seconds())
    df["time"] = df["td"]
    df = df.set_index("time")
    df.index = pd.to_datetime(df.index, unit = "s")
    
    ## Acceleration Resultant of acceleration
    df["AA"] = np.sqrt(df["x"]**2 + df["y"]**2 + df["z"]**2)
    df = df[["td", "x", "y", "z", "AA"]]
    return df