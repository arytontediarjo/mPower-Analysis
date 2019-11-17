"""
Functional script for cleaning data that will be used for analysis

Cleanups done:
    -> Removing Error caused from feature extraction, which includes empty pathfiles, empty acceleration data,
        data that does not contain userAcceleration values (Android)
    -> Removing duplicated records, healthCodes that have the same data accross two different recordIds
    -> Joining outbound and return user acceleration as one feature on mPower Version 1 
    -> Rename feature name more consistently into {coordinate}.{name of the feature} e.g. x.number_of_steps
"""


import sys
import warnings
import pandas as pd
import numpy as np
import synapseclient as sc
from utils import save_data_to_synapse, fix_column_name
from synapseclient import Entity, Project, Folder, File, Link, Activity
import time
import os
from datetime import datetime
warnings.simplefilter("ignore")


"""
GLOBAL VARIABLES
"""
DEMOGRAPHIC_TABLE_V1 = "syn8381056"
DEMOGRAPHIC_TABLE_V2 = "syn15673379"
WALKING_TABLE_V1     = "syn21111818"
WALKING_TABLE_V2     = "syn21113231"
PASSIVE_TABLE        = "syn21114136"
BALANCE_TABLE_V1     = "syn21028418"
BALANCE_TABLE_V2     = "syn21018245"
DATA_PARENT_ID       = "syn21024857"
SCRIPT_PARENT_ID     = "syn20987850"


syn = sc.login()

"""
Function to get demographic data
params: Version: Which mPower version 
        demographic table: Which demographic table to query from
returns: demographic data 
"""
def get_demographic_data(version, demographic_table_id):
    
    ### Version 1 data gathered from .tsv assays ###
    if version == "V1":
        demographic_entity = syn.get(demographic_table_id)
        demographic_data   = pd.read_csv(demographic_entity["path"], sep = "\t")[["healthCode", "PD", 
                                                                                  "gender", "age", "education"]]
        demographic_data["PD"] = demographic_data["PD"].map({True:1, False:0})
        demographic_data["gender"] = demographic_data["gender"].apply(lambda x: x.lower())
    
    ### Version 2 data gathered from table entity ###
    else:
        demographic_data   = syn.tableQuery("SELECT distinct(healthCode) as healthCode, \
                                             diagnosis as PD, sex as gender, birthYear from {} \
                                             where dataGroups NOT LIKE '%test_user%'".format(demographic_table_id))
        demographic_data   = demographic_data.asDataFrame().drop_duplicates("healthCode", keep = "first").reset_index(drop = True)
        demographic_data   = demographic_data[demographic_data["PD"] != "no_answer"] 
        demographic_data["PD"] = demographic_data["PD"].map({"parkinsons":1, "control":0})
        demographic_data["age"] = demographic_data["birthYear"].apply(lambda year: datetime.now().year - year)
        demographic_data = demographic_data[["healthCode", "PD", "gender", "age"]]
    return demographic_data
    

"""
Function to clean a data
params: version: Which version of the app
        demographic_table_id: synID of demographic table to get additional data
        walking_table_id: synID of the file entity of featurized walking data
        script_parent_id: synID of the script folders that refers back to this script and save it to that folder
        output_filename: name of the file user wants to output the cleaned dataset as
returns: a saved script and cleaned dataset in Synapse
"""
def clean_data(version, demographic_table_id, 
               walking_table_id, 
               data_parent_id,
               script_parent_id, 
               output_filename):
    
    ### Get demographics data ###
    demographic_data = get_demographic_data(version, demographic_table_id)
    
    
    ### Get data from raw gait data files ###
    entity = syn.get(walking_table_id)
    data   = pd.read_csv(entity["path"], index_col = 0)
    
    ### Drop duplicates of activities that contains the same data
    ### by removing same healthcodes that has repeated data creation date ###
    data.drop_duplicates(subset=['healthCode', 'createdOn'], keep = "first", inplace = True)
    
    
    ### rename columns accordingly and concatenate outbound and return data for version one, 
    ### while version 2 does not require extra concatenation as all walking data is consolidated
    ### into one ###
    if version == "V1":
        data_return   = data[[feature for feature in data.columns if "outbound" not in feature]]
        data_outbound = data[[feature for feature in data.columns if "return" not in feature]]
        data = pd.concat([fix_column_name(data_outbound), fix_column_name(data_return)])
    else:
        data = fix_column_name(data)
    
    ### remove empty cells that contains empty pdkit features ### 
    data = (data[data["phoneInfo"].str.contains("iPhone")]) \
                            [(data != "#NULL_FROM_PDKIT").all(axis = 1)]
                            
    ### change dtype to float64 ### 
    data[[_ for _ in data.columns if "feature" in _]] = \
    data[[_ for _ in data.columns if "feature" in _]].apply(pd.to_numeric)
    
    
    ### reset indexing of data, and remove redundant duration data gathered 
    ### from AWS data pipeline ###
    data.reset_index(drop = True, inplace = True)
    data.drop(["y.duration", "z.duration", "AA.duration"], axis = 1, inplace = True) 
    data.rename({"x.duration": "duration"}, axis = 1, inplace = True)
    
    ### Merge with demographic data to have more metadata ###
    data = pd.merge(data, demographic_data, 
                    how = "inner", on = "healthCode")
    
    
    ### save script to synapse and save cleaned dataset to synapse ###
    save_data_to_synapse(data, __file__, 
                    walking_table_id,
                    data_parent_id,
                    script_parent_id, 
                    output_filename)
    

"""
main function
iteratively cleaned all raw featurized datasets given their demographics data, filename etc.
"""
def main():
    cleaned_PDKIT_MPV1 = clean_data("V1", DEMOGRAPHIC_TABLE_V1, 
                                    WALKING_TABLE_V1, DATA_PARENT_ID,
                                    SCRIPT_PARENT_ID, "cleaned_pdkit_mpv1.csv")
    cleaned_PDKIT_MPV2 = clean_data("V2", DEMOGRAPHIC_TABLE_V2, 
                                    WALKING_TABLE_V2, DATA_PARENT_ID,
                                    SCRIPT_PARENT_ID, "cleaned_pdkit_mpv2.csv")
    # cleaned_SFM_MPV1 = clean_data("V1", DEMOGRAPHIC_TABLE_V1, 
    #                               BALANCE_TABLE_V1, DATA_PARENT_ID,
    #                               SCRIPT_PARENT_ID, "cleaned_SFM_MPV1.csv")
    # cleaned_SFM_MPV2 = clean_data("V2", DEMOGRAPHIC_TABLE_V2, 
    #                               BALANCE_TABLE_V2, DATA_PARENT_ID,
    #                               SCRIPT_PARENT_ID, "cleaned_SFM_MPV2.csv")
    cleaned_PDKIT_PASSIVE= clean_data("V2", DEMOGRAPHIC_TABLE_V2, 
                                      PASSIVE_TABLE, DATA_PARENT_ID,
                                      SCRIPT_PARENT_ID, "cleaned_pdkit_passive.csv")
    
## Run Main Function ##
if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))

    
    
    
    