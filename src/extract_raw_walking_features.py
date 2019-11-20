"""
DESCRIPTION: Script for extracting walking features 
            features available: spectral flatness on balance tests and PDKIT features

Side Notes:
 - #ERROR string will be annotated on empty filepaths or empty dataframes
"""


import os
import pandas as pd
import numpy as np
import synapseclient as sc
from synapseclient import Entity, Project, Folder, File, Link, Activity
import time
import multiprocessing as mp
import warnings
import argparse
from utils.munging_utils import get_walking_synapse_table, get_healthcodes, \
                                check_children, parallel_func_apply, save_data_to_synapse
from utils.pdkit_feature_utils import pdkit_featurize, pdkit_normalize
from utils.sfm_feature_utils import sfm_featurize                                
warnings.simplefilter("ignore")



"""
Constants of table ID for query
"""
WALK_TABLE_V1      = "syn10308918"
WALK_TABLE_V2      = "syn12514611"
WALK_TABLE_PASSIVE = "syn17022539"
ELEVATE_MS         = "syn10278766"
GIT_URL = "https://github.com/arytontediarjo/mPower-Analysis/blob/master/PythonScripts/extract_raw_walking_features.py"



def read_args():
    """
    Function for parsing in argument given by client
    returns arguments parameter
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default= "V1", choices = ["V1", "V2", "PASSIVE", "EMS"],
                        help = "mpower version number (either V1 or V2)")
    parser.add_argument("--filename", default= "data.csv",
                        help = "Name for Output File")
    parser.add_argument("--num-cores", default= mp.cpu_count(),
                        help = "Number of Cores, negative number not allowed")
    parser.add_argument("--num-chunks", default= 250,
                        help = "Number of sample per partition, no negative number")
    parser.add_argument("--featurize", default= "spectral-flatness",
                        help = "Features choices: 'pdkit' or 'spectral-flatness' ")
    parser.add_argument("--filtered", action='store_true', 
                        help = "use case matched healthcodes?")
    parser.add_argument("--script", default= GIT_URL, 
                        help = "executed script")
    parser.add_argument("--data-parent-id", default = "syn20988708", 
                        help = "output data folders parent ids")
    parser.add_argument("--update", default = "store_true",
                        help = "user choice of update or start fresh")
    args = parser.parse_args()
    return args


def main():
    """
    Main function
    - Takes in arguments from user
    - Query walking and balance table from synapse
    - Parallelly process each features, to be featurized with choice of feature computation
    - Saves the raw data into synapse (synID: syn20988708) with provenance of source walking table id
      and script from github repository
    """ 
    ## Retrieve Arguments
    args = read_args()
    version = args.version
    output_filename = args.filename                      
    cores = int(args.num_cores)                     
    chunksize = int(args.num_chunks)                
    features = args.featurize                                                 
    is_filtered = args.filtered                                    
    data_parent_id = args.data_parent_id
    script = args.script
    is_update = args.update

    
    if version == "V1":
        source_table_id = WALK_TABLE_V1
    elif version == "V2":
        source_table_id = WALK_TABLE_V2
    elif version == "PASSIVE":
        source_table_id = WALK_TABLE_PASSIVE
    else:
        source_table_id = ELEVATE_MS   

    ### pseudo code ###
    """
    for the data parent id
    go through the children of files
    if file is already available 
    read the csv, get the recordIds
    subset the new query by available walking IDS
    then start featurizing on the data subset
    then concat the data 
    """

    ## process data ##
    data = get_walking_synapse_table(get_healthcodes(source_table_id, is_filtered), 
                                    source_table_id, version)
    print(data.shape)


    if is_update:
        print("update data")
        prev_stored_data, prev_recordId_list = check_children(data_parent_id, output_filename)
        print(prev_stored_data.shape)
    else:
        prev_stored_data= pd.DataFrame()
    

    ## only featurize new recordIds
    data = data[~data.isin(prev_recordId_list)]
    print(data.shape)
    
    ## condition on choosing which features
    print("Retrieving {} Features".format(features))
    if features == "spectral-flatness":
        data = parallel_func_apply(data, sfm_featurize, cores, chunksize)
    elif features == "pdkit":
        data = parallel_func_apply(data, pdkit_featurize, cores, chunksize)
        data = pdkit_normalize(data)
    print("parallelization process finished")
    data = data[[feat for feat in data.columns if ("path" not in feat) 
                 and ("0" not in feat)]]

    data = pd.concat([prev_stored_data, data]).reset_index(drop = True)
    
    
    ### save script to synapse and save cleaned dataset to synapse ###
    save_data_to_synapse(data            = data, 
                         output_filename = output_filename, 
                         data_parent_id  = data_parent_id,
                         used_script     = script,
                         source_table_id = source_table_id) 
                         
"""
Run main function and record the time of script runtime
"""
if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
