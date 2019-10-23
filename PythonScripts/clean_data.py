"""
Functional script for cleaning data that will be used for analysis

Cleanups done:
    -> Removing Error caused from feature extraction, which includes empty pathfiles, empty acceleration data,
        data that does not contain userAcceleration values (Android)
    -> Removing duplicate records, healthcodes that have the same data but different records
    -> Joining outbound and return user acceleration into one columns on mPowerV1
    -> Rename feature name more consistently into userAcc_feature_{orientation X, Y, Z}.{name of feature}
"""


import sys
import warnings
import pandas as pd
import numpy as np
import synapseclient as sc
from utils import get_script_id
from synapseclient import Entity, Project, Folder, File, Link, Activity
import time
import os
warnings.simplefilter("ignore")

DEMOGRAPHIC_TABLE_V1 = "syn8381056"
DEMOGRAPHIC_TABLE_V2 = "syn15673379"
WALKING_TABLE_V1     = "syn21021435"
WALKING_TABLE_V2     = "syn21018127"
PASSIVE_TABLE        = "syn21028519"
BALANCE_TABLE_V1     = "syn21028418"
BALANCE_TABLE_V2     = "syn21018245"
DATA_PARENT_ID       = "syn21024857"
SCRIPT_PARENT_ID     = "syn20987850"
syn = sc.login()


def fix_column_name(data):
    for feature in filter(lambda x: "feature" in x, data.columns): 
        data  = data.rename({feature: "userAccel{}"\
                            .format(feature.split("features")[1])}, axis = 1)
    return data


def clean_data(version, demographic_table_id, walking_table_id, script_parent_id, filename):
    if version == "V1":
        ## demographic data ## 
        demographic_entity = syn.get(demographic_table_id)
        demographic_data   = pd.read_csv(demographic_entity["path"], sep = "\t")[["healthCode", "PD", "gender", "age"]]
        demographic_data["PD"] = demographic_data["PD"].map({True:1, False:0})
    else:
        demographic_data   = syn.tableQuery("SELECT distinct(healthCode) as healthCode, \
                                             diagnosis as PD, sex as gender from {}".format(demographic_table_id))
        demographic_data   = demographic_data.asDataFrame().drop_duplicates("healthCode", keep = "first").reset_index(drop = True)
        demographic_data   = demographic_data[demographic_data["PD"] != "no_answer"] 
        demographic_data["PD"] = demographic_data["PD"].map({"parkinsons":1, "control":0})
    entity = syn.get(walking_table_id)
    data   = pd.read_csv(entity["path"], index_col = 0)
    data = (data[data["phoneInfo"].str.contains("iPhone")]) \
                            [(data != "#ERROR").all(axis = 1)]
    data.drop_duplicates(subset=['healthCode', 'createdOn'], keep = "first", inplace = True)
    data[[_ for _ in data.columns if "feature" in _]] = \
    data[[_ for _ in data.columns if "feature" in _]].apply(pd.to_numeric)
    
    data.reset_index(drop = True, inplace = True)
    data = pd.merge(data, demographic_data, 
                    how = "inner", on = "healthCode")
    
    ## rename columns and concat 
    if version == "V1":
        data_return   = data[[feature for feature in data.columns if "outbound" not in feature]]
        data_outbound = data[[feature for feature in data.columns if "return" not in feature]]
        data = pd.concat([fix_column_name(data_outbound), fix_column_name(data_return)])
    else:
        data = fix_column_name(data)
    
    ## store to synapse    
    path_to_script = os.path.join(os.getcwd(), __file__)
    output_filename = os.path.join(os.getcwd(), filename)
    data = data.to_csv(output_filename)
    new_file = File(path = output_filename, parentId = DATA_PARENT_ID)
    new_file = syn.store(new_file)
    # syn.setProvenance(new_file, 
    #                   activity = Activity(used = [walking_table_id], 
    #                                       executed = get_script_id(syn, __file__, script_parent_id)))
                   
    os.remove(output_filename)
    

    
    

def main():
    cleaned_PDKIT_MPV1 = clean_data("V1", DEMOGRAPHIC_TABLE_V1, 
                                    WALKING_TABLE_V1, SCRIPT_PARENT_ID, "cleaned_PDKIT_MPV1.csv")
    cleaned_PDKIT_MPV2 = clean_data("V2", DEMOGRAPHIC_TABLE_V2, 
                                    WALKING_TABLE_V2, SCRIPT_PARENT_ID, "cleaned_PDKIT_MPV2.csv")
    cleaned_SFM_MPV1 = clean_data("V1", DEMOGRAPHIC_TABLE_V1, 
                                  BALANCE_TABLE_V1, SCRIPT_PARENT_ID, "cleaned_SFM_MPV1.csv")
    cleaned_SFM_MPV2 = clean_data("V2", DEMOGRAPHIC_TABLE_V2, 
                                  BALANCE_TABLE_V2, SCRIPT_PARENT_ID, "cleaned_SFM_MPV2.csv")
    cleaned_PDKIT_PASSIVE= clean_data("V2", DEMOGRAPHIC_TABLE_V2, 
                                      PASSIVE_TABLE, SCRIPT_PARENT_ID, "cleaned_PDKIT_PASSIVE.csv")
    
## Run Main Function ##
if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))

    
    
    
    