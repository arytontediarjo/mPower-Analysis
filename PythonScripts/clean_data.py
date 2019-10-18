import sys
import warnings
import pandas as pd
import numpy as np
import synapseclient as sc
warnings.simplefilter("ignore")

DEMOGRAPHIC_TABLE_V1 = "syn8381056"
DEMOGRAPHIC_TABLE_V2 = "syn15673379"
WALKING_TABLE_V1     = "syn21021435"
WALKING_TABLE_V2     = "syn21018127"
BALANCE_TABLE_V1     = "syn21022189"
BALANCE_TABLE_V2     = "syn21018245"


def clean_data(version, demographic_table, walking_table, balance_table):
    if version == "V1":
        ## demographic data ## 
        demographic_entity = syn.get(demographic_table)
        demographic_data   = pd.read_csv(demographic_entity["path"], sep = "\t")[["healthCode", "PD"]]
    else:
        demographic_data   = syn.tableQuery("SELECT distinct(healthCode) as healthCode, \
                                             diagnosis as PD from {}".format(demographic_table))
        demographic_data   = demographic_data.asDataFrame().drop_duplicates("healthCode", keep = "first").reset_index(drop = True)
        demographic_data   = demographic_data[demographic_data["PD"] != "no_answer"] 
    demographic_data["PD"] = demographic_data["PD"].map({"parkinsons":1, "control":0})
    pdkit_entity           = syn.get(walking_table)
    sfm_entity             = syn.get(balance_table)
    
    ### read data ###
    pdkit_data = pd.read_csv(pdkit_entity["path"], index_col = 0)
    ### remove errors, query only iPhone Data ###
    pdkit_data = (pdkit_data[pdkit_data["phoneInfo"].str.contains("iPhone")]) \
                            [(pdkit_data != "#ERROR").all(axis = 1)] 
    ### remove duplicates of multiple similar recordIds ###
    pdkit_data.drop_duplicates(subset=['healthCode', 'createdOn'], keep = "first", inplace = True)
    ### convert type to numeric ###
    pdkit_data[[_ for _ in pdkit_data.columns if "feat" in _]] = \
    pdkit_data[[_ for _ in pdkit_data.columns if "feat" in _]].apply(pd.to_numeric)

    ### join data ###
    pdkit_data.reset_index(drop = True, inplace = True)
    pdkit_data = pd.merge(pdkit_data, demographic_data, 
                               how = "inner", on = "healthCode")

    ## get sfm data ## 
    sfm_data = pd.read_csv(sfm_entity["path"], index_col = 0)
    sfm_data = (sfm_data[sfm_data["phoneInfo"].str.contains("iPhone")]) \
                            [(sfm_data != "#ERROR").all(axis = 1)] 
    ### remove duplicates of multiple similar recordIds ###
    sfm_data.drop_duplicates(subset=['healthCode', 'createdOn'], keep = "first", inplace = True)
    sfm_data[[_ for _ in sfm_data.columns if "sfm" in _]] = \
    sfm_data[[_ for _ in sfm_data.columns if "sfm" in _]].apply(pd.to_numeric)
    sfm_data.reset_index(drop = True, inplace = True)            
    sfm_data = pd.merge(sfm_data, demographic_data, 
                             how = "inner", on = "healthCode")
    
    return pdkit_data, sfm_data

def main():
    cleaned_mpV1_pdkit_data,  cleaned_mpV2_sfm_data = \
        clean_data("V1", DEMOGRAPHIC_TABLE_V1, WALKING_TABLE_V1, BALANCE_TABLE_V1)
    cleaned_mpV2_pdkit_data,  cleaned_mpV2_sfm_data = \
        clean_data("V2", DEMOGRAPHIC_TABLE_V2, WALKING_TABLE_V2, BALANCE_TABLE_V2)
    path_to_script = os.path.join(os.getcwd(), __file__)
    output_filename = os.path.join(os.getcwd(), filename)
    data = data.to_csv(output_filename)
    new_file = File(path = output_filename, parentId = data_parent_id)
    new_file = syn.store(new_file)
    os.remove(output_filename)
    syn.setProvenance(new_file, 
                      activity = Activity(used = synId, 
                                          executed = get_script_id(syn, __file__, "syn20987850")))
    
    

    
    
    
    