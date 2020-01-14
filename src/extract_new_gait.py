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
    data_return   = query_data_v1[[feature for feature in query_data_v1.columns if "outbound" not in feature]]
    data_outbound = query_data_v1[[feature for feature in query_data_v1.columns if "return" not in feature]]
    query_data_v1 = pd.concat([data_outbound, data_return])## combine return and outbound                   
    arr_outbound = query_data_v1["deviceMotion_walking_outbound.json.items_pathfile"].dropna()
    arr_return = query_data_v1["deviceMotion_walking_return.json.items_pathfile"].dropna()
    query_data_v1["walk_motion.json_pathfile"] = pd.concat([arr_outbound, arr_return])

    ## healthcode from version 2 ## 
    hc_arr_v2 = (matched_demographic["healthCode"][matched_demographic["version"] == "mpower_v2"].unique())
    query_data_v2 = query.get_walking_synapse_table(syn, 
                                                    "syn12514611", 
                                                    "MPOWER_V2", 
                                                    healthCodes = hc_arr_v2)
    data = pd.concat([query_data_v1, query_data_v2]).reset_index(drop = True)                                             
    
    
    
    ## create pdkit ##
    # pdkit_data = query.parallel_func_apply(data, gproc.pdkit_featurize_wrapper, 16, 250)
    # pdkit_data = pdkit_data[pdkit_data["gait.pdkit_features"] != "#ERROR"]
    # pdkit_data = query.normalize_list_dicts_to_dataframe_rows(pdkit_data, ["gait.pdkit_features"])
    # pdkit_data = gproc.subset_data_non_zero_runs(data = pdkit_data, zero_runs_cutoff = 5)
    # ### save data to synapse ##
    # query.save_data_to_synapse(syn = syn, 
    #                         data = pdkit_data, 
    #                         output_filename = "new_pdkit_features_matched.csv",
    #                         data_parent_id = "syn20816722")        
    # print("Saved walking data")                                                
    
    ## create rotation file through multiprocessing jobs ## 
    rotation_data = query.parallel_func_apply(data, gproc.rotation_featurize_wrapper, 16, 250) 
    rotation_data = rotation_data[rotation_data["gait.rotational_features"] != "#ERROR"]
    rotation_data = query.normalize_list_dicts_to_dataframe_rows(rotation_data, ["gait.rotational_features"])
    
    metadata_feature = ['recordId', 'healthCode','appVersion', 'phoneInfo', 'createdOn']
    rotation_feature = [feat for feat in rotation_data.cols if "rotation." in feat]
    feature_cols = metadata_feature + rotation_feature

    ## save data to synapse ##
    query.save_data_to_synapse(syn = syn, 
                            data = rotation_data[feature_cols], 
                            output_filename = "new_rotational_features_matched.csv",
                            data_parent_id = "syn20816722")

    print("Saved rotation data") 


if __name__ ==  '__main__': 
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))