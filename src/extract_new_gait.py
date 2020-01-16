import pdkit
import synapseclient as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import utils.query_utils as query
import utils.gait_feature_prototype_utils as gproc
import synapseclient as sc
from sklearn import metrics
import time
warnings.simplefilter("ignore")

def clean_gait_mpower_dataset(data, filepath_colname, test_type):
    metadata = ["appVersion", "phoneInfo", "healthCode", "recordId", "createdOn"]
    data = data[[feature for feature in data.columns if \
                            (filepath_colname in feature) or \
                            feature in metadata]]\
                            .rename({filepath_colname: "gait.json_pathfile"}, 
                                       axis = 1)
    data["test_type"] = test_type
    return data



def main():
    syn = sc.login()

    ## gather matched demographics ## 
    matched_demographic = query.get_file_entity(syn, "syn21482502")

    ## healthcode from version 1 ##
    hc_arr_v1 = (matched_demographic["healthCode"][matched_demographic["version"] == "mpower_v1"].unique())
    query_data_v1 = query.get_walking_synapse_table(syn, 
                                                    "syn10308918", 
                                                    "MPOWER_V1", 
                                                    healthCodes = hc_arr_v1)
    ## healthcode from version 2 ## 
    hc_arr_v2 = (matched_demographic["healthCode"][matched_demographic["version"] == "mpower_v2"].unique())
    query_data_v2 = query.get_walking_synapse_table(syn, 
                                                    "syn12514611", 
                                                    "MPOWER_V2", 
                                                    healthCodes = hc_arr_v2)
    
    data_outbound_v1 = clean_gait_mpower_dataset(query_data_v1, 
                                                "deviceMotion_walking_outbound.json.items_pathfile",
                                                "walking")
    

    data_return_v1 = clean_gait_mpower_dataset(query_data_v1, 
                                                "deviceMotion_walking_return.json.items_pathfile",
                                                "walking")

    data_balance_v1 = clean_gait_mpower_dataset(query_data_v1, 
                                                "deviceMotion_walking_rest.json.items_pathfile",
                                                "balance")

    data_walking_v2 = clean_gait_mpower_dataset(query_data_v2, 
                                                "walk_motion.json_pathfile",
                                                "walking")


    data_balance_v2 = clean_gait_mpower_dataset(query_data_v2, 
                                                "balance_motion.json_pathfile",
                                                "balance")

    data = pd.concat([data_outbound_v1, 
                  data_return_v1, 
                  data_balance_v1, 
                  data_walking_v2, 
                  data_balance_v2]).reset_index(drop = True)

    print("dataset combined, total rows are %s" %data.shape[0])
    
    
    ## create pdkit ##
    walk_data = query.parallel_func_apply(data, gproc.walk_featurize_wrapper, 16, 250)
    walk_data = walk_data[walk_data["gait.walk_features"] != "#ERROR"]
    walk_data = query.normalize_list_dicts_to_dataframe_rows(walk_data, ["gait.walk_features"])
    metadata_feature = ['recordId', 'healthCode','appVersion', 'phoneInfo', 'createdOn', 'test_type']
    walking_feature = [feat for feat in walk_data.columns if "walking." in feat]
    features = metadata_feature + walking_feature

    ### save data to synapse ##
    query.save_data_to_synapse(syn = syn, 
                            data = walk_data[features], 
                            output_filename = "new_walk_features_matched.csv",
                            data_parent_id = "syn20816722")        
    print("Saved walking data")                                                
    
    ## create rotation file through multiprocessing jobs ## 
    rotation_data = query.parallel_func_apply(data, gproc.rotation_featurize_wrapper, 16, 250) 
    rotation_data = rotation_data[rotation_data["gait.rotational_features"] != "#ERROR"]
    rotation_data = query.normalize_list_dicts_to_dataframe_rows(rotation_data, ["gait.rotational_features"])
    metadata_feature = ['recordId', 'healthCode','appVersion', 'phoneInfo', 'createdOn', 'test_type']
    rotation_feature = [feat for feat in rotation_data.columns if "rotation." in feat]
    features = metadata_feature + rotation_feature

    ## save data to synapse ##
    query.save_data_to_synapse(syn = syn, 
                            data = rotation_data[features], 
                            output_filename = "new_rotational_features_matched.csv",
                            data_parent_id = "syn20816722")

    print("Saved rotation data") 


if __name__ ==  '__main__': 
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))