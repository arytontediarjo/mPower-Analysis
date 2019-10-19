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
BALANCE_TABLE_V1     = "syn21022189"
BALANCE_TABLE_V2     = "syn21018245"
DATA_PARENT_ID       = "syn21024857"
SCRIPT_PARENT_ID     = "syn20987850"
syn = sc.login()


def clean_data(version, demographic_table_id, walking_table_id, filename):
    if version == "V1":
        ## demographic data ## 
        demographic_entity = syn.get(demographic_table_id)
        demographic_data   = pd.read_csv(demographic_entity["path"], sep = "\t")[["healthCode", "PD"]]
    else:
        demographic_data   = syn.tableQuery("SELECT distinct(healthCode) as healthCode, \
                                             diagnosis as PD from {}".format(demographic_table_id))
        demographic_data   = demographic_data.asDataFrame().drop_duplicates("healthCode", keep = "first").reset_index(drop = True)
        demographic_data   = demographic_data[demographic_data["PD"] != "no_answer"] 
    demographic_data["PD"] = demographic_data["PD"].map({"parkinsons":1, "control":0})
    entity = syn.get(walking_table_id)
    data   = pd.read_csv(entity["path"], index_col = 0)
    data = (data[data["phoneInfo"].str.contains("iPhone")]) \
                            [(data != "#ERROR").all(axis = 1)]
    data.drop_duplicates(subset=['healthCode', 'createdOn'], keep = "first", inplace = True)
    data[[_ for _ in data.columns if "feat" in _]] = \
    data[[_ for _ in data.columns if "feat" in _]].apply(pd.to_numeric)
    
    data.reset_index(drop = True, inplace = True)
    data = pd.merge(data, demographic_data, 
                    how = "inner", on = "healthCode")
    
    path_to_script = os.path.join(os.getcwd(), __file__)
    output_filename = os.path.join(os.getcwd(), filename)
    data = data.to_csv(output_filename)
    new_file = File(path = output_filename, parentId = DATA_PARENT_ID)
    new_file = syn.store(new_file)
    syn.store(new_file)
                    #   activity = Activity(used = walking_table_id, 
                    #                       executed = get_script_id(syn, __file__, SCRIPT_PARENT_ID)))
    os.remove(output_filename)
    
    

def main():
    cleaned_PDKIT_MPV1 = clean_data("V1", DEMOGRAPHIC_TABLE_V1, WALKING_TABLE_V1, "cleaned_PDKIT_MPV1.csv")
    cleaned_PDKIT_MPV2 = clean_data("V2", DEMOGRAPHIC_TABLE_V2, WALKING_TABLE_V2, "cleaned_PDKIT_MPV2.csv")
    cleaned_SFM_MPV1 = clean_data("V1", DEMOGRAPHIC_TABLE_V1, BALANCE_TABLE_V1, "cleaned_SFM_MPV1.csv")
    cleaned_SFM_MPV2 = clean_data("V2", DEMOGRAPHIC_TABLE_V2, BALANCE_TABLE_V2, "cleaned_SFM_MPV2.csv")
    
## Run Main Function ##
if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))

    
    
    
    