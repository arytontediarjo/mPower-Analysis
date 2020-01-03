import sys
import json 
import os
import ast
import pandas as pd
import numpy as np
import synapseclient as sc
from synapseclient import (Entity, Project, Folder, File, Link, Activity)
import multiprocessing as mp
from multiprocessing import Pool


def get_walking_synapse_table(syn, 
                            table_id, 
                            version, 
                            healthCodes = None, 
                            recordIds = None):
    """
    Query synapse walking table entity 
    parameters:  
    `syn`         : synapse object,             
    `table_id`    : id of table entity,
    `version`     : version number (args (string) = ["V1", "V2", "EMS", "Passive"])
    `healthcodes` : list of healthcodes
    `recordIDs`   : list of recordIds
    
    returns: a dataframe of recordIds and their respective metadata, alongside their filehandleids and filepaths
             empty filepath will be annotated as "#ERROR" on the dataframe
    """
    print("Querying %s Data" %version)
    if not isinstance(recordIds, type(None)):
        if not isinstance(recordIds, list):
            recordIds = list(recordIds)
        recordId_subset = "({})".format([i for i in recordIds]).replace("[", "").replace("]", "")
        query = syn.tableQuery("select * from {} WHERE recordId in {}".format(table_id, recordId_subset))
    else:
        if not isinstance(healthCodes, list):
            healthCodes = list(healthCodes)
        healthCode_subset = "({})".format([i for i in healthCodes]).replace("[", "").replace("]", "")
        query = syn.tableQuery("select * from {} WHERE healthCode in {}".format(table_id, healthCode_subset))
    data = query.asDataFrame()

    ## unique table identifier in mpowerV1 and EMS synapse table
    if (version == "MPOWER_V1") or (version == "MS_ACTIVE"):
        feature_list = [_ for _ in data.columns if ("deviceMotion" in _) and ("rest" not in _)]
    ## unique table identifier in mpowerV2 and passive data
    else:
        feature_list = [_ for _ in data.columns if ("json" in _)]
    ## download columns that contains walking data based on the logical condition
    file_map = syn.downloadTableColumns(query, feature_list)
    
    dict_ = {}
    dict_["file_handle_id"] = []
    dict_["file_path"] = []
    for k, v in file_map.items():
        dict_["file_handle_id"].append(k)
        dict_["file_path"].append(v)
    filepath_data = pd.DataFrame(dict_)
    data = data[["recordId", "healthCode", 
                "appVersion", "phoneInfo", 
                "createdOn"] + feature_list]
    filepath_data["file_handle_id"] = filepath_data["file_handle_id"].astype(float)
    
    ### Join the filehandles with each acceleration files ###
    for feat in feature_list:
        data[feat] = data[feat].astype(float)
        data = pd.merge(data, filepath_data, 
                        left_on = feat, 
                        right_on = "file_handle_id", 
                        how = "left")
        data = data.rename(columns = {feat: "{}_path_id".format(feat), 
                                    "file_path": "{}_pathfile".format(feat)}).drop(["file_handle_id"], axis = 1)
    ## Empty Filepaths on synapseTable ##
    data = data.fillna("#ERROR") 
    return data


def get_sensor_ts(filepath, sensor): 
    """
    Function to get accelerometer data given a filepath,
    will adjust to different table entity versions accordingly by 
    extracting specific keys in json pattern. 
    Empty filepaths will be annotated with "#ERROR"

    parameters : 
    `filepath` : filepaths of given data

    return a tidied version of the dataframe that contains a time-index dataframe (timestamp), 
    time differences (td), (x, y, z, AA) user acceleration (non-g)
    """

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
            data = data[data["sensorType"] == sensor]
            data = clean_accelerometer_data(data)
        except:
            return "#ERROR"
        return data[["td","x", "y", "z", "AA"]]
        
    ## userAcceleration from mpowerV1
    elif ("userAcceleration" in data.columns):
        data = data[["timestamp", sensor]]
        data["x"] = data[sensor].apply(lambda x: x["x"])
        data["y"] = data[sensor].apply(lambda x: x["y"])
        data["z"] = data[sensor].apply(lambda x: x["z"])
        data = data.drop([sensor], axis = 1)
        data = clean_accelerometer_data(data)
        return data[["td","x", "y", "z", "AA"]]
    

def clean_accelerometer_data(data):
    """
    Generalized function to clean accelerometer data to
    a desirable format 

    parameter: 
    `data`: time-series dataset

    returns index (datetimeindex), td (float64), 
            x (float64), y (float64), z (float64),
            AA (float64) dataframe
        
    """
    data = data.dropna(subset = ["x", "y", "z"])
    date_series = pd.to_datetime(data["timestamp"], unit = "s")
    data["td"] = date_series - date_series.iloc[0]
    data["td"] = data["td"].apply(lambda x: x.total_seconds())
    data["time"] = data["td"]
    data = data.set_index("time")
    data.index = pd.to_datetime(data.index, unit = "s")
    data["AA"] = np.sqrt(data["x"]**2 + data["y"]**2 + data["z"]**2)
    data = data.sort_index()
    
    ## check if datetime index is sorted ##
    if all(data.index[:-1] <= data.index[1:]):
        return data 
    else:
        sys.exit('Time Series File is not Sorted')


def open_filepath(filepath):
    """
    General Function to open a filepath 
    parameter: a filepath
    return: pandas dataframe of the respective filepath
    """
    with open(filepath) as f:
        json_data = f.read()
        data = pd.DataFrame(json.loads(json_data))
    return data


def get_healthcodes(syn, table_id):
    """
    Function to get healthCodes in python list format
    parameter:  syn: syn object,
                synId: table that user want to query from,    
    returns list of healthcodes
    """
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
    
    
def save_data_to_synapse(syn,
                        data, 
                        output_filename,
                        data_parent_id, 
                        used_script = None,
                        source_table_id = None,
                        remove = True): 

    """
    Function to save data to synapse given a parent id, used script, 
    and source table where the query was sourced
    params: 
    `syn`              = synapse object        
    `data`             = tabular data, script or notebook 
    `output_filename`  = the name of the output file 
    `data_parent_id`   = the parent synid where data will be stored 
    `used_script`      = git repo url that produces this data (if available)
    `source_table_id`  = list of source of where this data is produced (if available) 
    `remove`           = remove data after saving, generally used for csv data 

    returns stored file entity in Synapse Database
    """
    ## path to output filename for reference ##
    path_to_output_filename = os.path.join(os.getcwd(), output_filename)
        
    ## save the script to synapse ##
    if isinstance(data, pd.DataFrame):
        data = data.to_csv(path_to_output_filename)
    
    ## create new file instance and set up the provenance
    new_file = File(path = path_to_output_filename, parentId = data_parent_id)
        
    ## instantiate activity object
    act = Activity()
    if source_table_id is not None:
        act.used(source_table_id)
    if used_script is not None:
        act.executed(used_script)
        
    ## store to synapse ## 
    new_file = syn.store(new_file, activity = act)           
        
    ## remove the file ##
    if remove:
        os.remove(path_to_output_filename)

  
def normalize_dict_features(data, feature):
    """
    Function to normalize column that conatins dictionaries different columns
    parameter: 
    `data`    : the data itself           
    `feature` : the target feature
    
    returns a normalized dataframe with column containing dictionary normalized
    """    
    normalized_data = data[feature].map(lambda x: x if isinstance(x, dict) else "#ERROR") \
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



def get_file_entity(syn, synid):
    """
    Get file entity and turn it into pandas csv
    returns pandas dataframe 
    parameters:
    `syn`: a syn object
    `synid`: synid of file entity

    returns pandas dataframe
    """
    entity = syn.get(synid)
    if (".tsv" in entity["name"]):
        separator = "\t"
    else:
        separator = ","
    data = pd.read_csv(entity["path"],index_col = 0, sep = separator)
    return data



def parallel_func_apply(df, func, no_of_processors, chunksize):
    """
    Function for parallelization
    parameter: 
    `df`               = dataset           
    `func`             = function for data transformation
    `no_of_processors` = number of processors to transform the data
    `chunksize`        = number of chunk partition 
    
    return: featurized dataframes
    """
    df_split = np.array_split(df, chunksize)
    print("Currently running on {} processors".format(no_of_processors))
    pool = Pool(no_of_processors)
    map_values = pool.map(func, df_split)
    df = pd.concat(map_values)
    pool.close()
    pool.join()
    return df

def check_children(syn, data_parent_id, filename):
    """
    Function to check if file is already available
    if file is available, get all the recordIds and all the file
    parameter: 
    `syn` = syn object           
    `data_parent_id` = the parent folder
    `output_filename` = the filename
    
    returns previously stored dataframe
    """
    prev_stored_data = pd.DataFrame()
    prev_recordId_list = []
    for children in syn.getChildren(parent = data_parent_id):
            if children["name"] == filename:
                prev_stored_data_id = children["id"]
                prev_stored_data = get_file_entity(syn, prev_stored_data_id)
    return prev_stored_data


    