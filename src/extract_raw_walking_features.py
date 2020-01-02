"""
SCRIPT FOR EXTRACTING GAIT FEATURES ON ALL VERSION FROM SYNAPSE TABLE
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import pandas as pd
import numpy as np
import synapseclient as sc
from synapseclient import (Entity, Project, Folder, File, Link, Activity)
import time
import multiprocessing as mp
import warnings
import argparse
from utils.query_utils import (get_walking_synapse_table, get_healthcodes, \
                                check_children, parallel_func_apply, save_data_to_synapse)
from utils.pdkit_feature_utils import (pdkit_featurize, normalize_pdkit_features)
from utils.sfm_feature_utils import sfm_featurize 
warnings.simplefilter("ignore")                               



"""
Constants of table ID for query
"""
WALK_TABLE_V1         = "syn10308918"
WALK_TABLE_V2         = "syn12514611"
WALK_TABLE_PASSIVE    = "syn17022539"
ELEVATE_MS_ACTIVE     = "syn10278766"
ELEVATE_MS_PASSIVE    = "syn10651116"
RAW_DATA_PARENT_ID    = "syn20988708" 
GIT_URL = "https://github.com/arytontediarjo/mPower-Analysis/blob/master/PythonScripts/extract_raw_walking_features.py"



def read_args():
    """
    Function for parsing in argument given by client
    returns argument parameter
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default= "MPOWER_V1", choices = ["MPOWER_V1", "MPOWER_V2", "MPOWER_PASSIVE", "MS_ACTIVE"],
                        help = "query mpower version number (MPOWER_V1, MPOWER_V2, MPOWER_PASSIVE or MS_ACTIVE)")
    parser.add_argument("--filename", default= "data.csv",
                        help = "name of the output file per query")
    parser.add_argument("--num-cores", default= mp.cpu_count(),
                        help = "Number of Cores, negative number not allowed")
    parser.add_argument("--num-chunks", default= 250,
                        help = "Number of sample per partition, no negative number")
    parser.add_argument("--featurize", default= "spectral-flatness", choices = ["pdkit", "sfm"],
                        help = "Featurization choices: 'pdkit', 'sfm'")
    parser.add_argument("--update", action = "store_true",
                        help = "If specified, will update data based on new recordIds, otherwise start fresh")
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

    syn = sc.login()
    ## Retrieve Arguments
    args = read_args() 
    if args.version == "MPOWER_V1":
        source_table_id = WALK_TABLE_V1
    elif args.version == "MPOWER_V2":
        source_table_id = WALK_TABLE_V2
    elif args.version == "MPOWER_PASSIVE":
        source_table_id = WALK_TABLE_PASSIVE
    elif args.version == "MS_ACTIVE":
        source_table_id = ELEVATE_MS_ACTIVE
    data = get_walking_synapse_table(syn = syn, 
                                    table_id = source_table_id, 
                                    version = args.version, 
                                    healthCodes = get_healthcodes(syn = syn, table_id = source_table_id))
    prev_stored_data   = pd.DataFrame()
    if args.update:
        print("\n#########  UPDATING DATA  ################\n")
        prev_stored_data = check_children(syn = syn,
                                        data_parent_id = RAW_DATA_PARENT_ID, 
                                        filename = args.filename)
        data = data[~data["recordId"].isin(prev_stored_data["recordId"].unique())]
        print(data.shape)
    if args.featurize == "sfm":
        print("processing spectral-flatness")
        data = parallel_func_apply(data, sfm_featurize, int(args.num_cores), int(args.num_chunks))
    elif args.featurize == "pdkit":
        print("processing pdkit")
        data = parallel_func_apply(data, pdkit_featurize, int(args.num_cores), int(args.num_chunks))
        data = normalize_pdkit_features(data)
    print("parallelization process finished")
    data = data[[feat for feat in data.columns if ("path" not in feat) and ("0" not in feat)]].reset_index(drop = True)
    data = pd.concat([prev_stored_data, data]).reset_index(drop = True)
    data = data.loc[:,~data.columns.duplicated()]
    save_data_to_synapse(syn = syn, 
                        data = data, 
                        output_filename = args.filename, 
                        data_parent_id  = RAW_DATA_PARENT_ID,
                        used_script     = GIT_URL,
                        source_table_id = source_table_id) 
                         
"""
Run main function and record runtime for each version query
"""
if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
