import sys
import os
import warnings
import pandas as pd
import numpy as np
import synapseclient as sc
from synapseclient import Entity, Project, Folder, File, Link, Activity
import time
from utils import get_sensor_types, \
                    get_units, get_synapse_table,  \
                    get_script_id, get_healthcodes, get_sensor_specs, \
                    normalize_feature
import argparse
import multiprocessing as mp
from multiprocessing import Pool
warnings.simplefilter("ignore")


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", default= "data.csv",
                        help = "Path for output results")
    parser.add_argument("--num-cores", default= 16,
                        help = "Number of Cores, negative number not allowed")
    parser.add_argument("--num-chunks", default= 250,
                        help = "Number of sample per partition, no negative number")
    parser.add_argument("--table-id", default= "syn7222425", ## mpower V1
                        help = "mpower gait table to query from")
    parser.add_argument("--filtered", action='store_true', 
                        help = "filter healthcodes")
    parser.add_argument("--script-parent-id", default= "syn20987850", 
                        help = "script folders parent ids")
    parser.add_argument("--data-parent-id", default = "syn20988708", 
                        help = "data folders parent ids")
    args = parser.parse_args()
    return args


"""
Function for parallelization
returns featurized dataframes
"""
def _parallelize_dataframe(df, func, no_of_processors, chunksize):
    ### split dataframe into 250 partitions ###
    df_split = np.array_split(df, chunksize)
    ### instantiate 16 processors as EC2 instance has 8 cores ###
    print("Currently running on {} processors".format(no_of_processors))
    pool = Pool(no_of_processors)
    ### map function into each pools ###
    map_values = pool.map(func, df_split)
    ### concatenate dataframe into one ###
    df = pd.concat(map_values)
    ### close pools
    pool.close()
    pool.join()
    return df

def _featurize(data):
    data["sensors_used_walking"] = data["walk_motion.json_pathfile"].apply(get_sensor_types)
    data["sensors_used_balance"] = data["balance_motion.json_pathfile"].apply(get_sensor_types)
    data["walking_units"] = data["walk_motion.json_pathfile"].apply(get_units)
    data["balance_units"] = data["balance_motion.json_pathfile"].apply(get_units)
    data["walking_specs"] = data["walk_motion.json_pathfile"].apply(get_sensor_specs)
    data["balance_specs"] = data["balance_motion.json_pathfile"].apply(get_sensor_specs)
    return data

def main(): 
    args              = read_args()            ## read arguments
    filename          = args.filename          ## name of the file
    synId             = args.table_id          ## which table to query from
    is_filtered       = args.filtered          ## filter the dataset
    data_parent_id    = args.data_parent_id    ## parent id of data
    script_parent_id  = args.script_parent_id  ## parent id of python scripts
    
    syn = sc.login()
    data = get_synapse_table(syn, get_healthcodes(syn, synId, is_filtered), synId)
    data = _parallelize_dataframe(data, _featurize, 16, 250)
    data = normalize_feature(data, "walking_specs")
    data = normalize_feature(data, "balance_specs")
    
    ## store data and script ##
    path_to_script = os.path.join(os.getcwd(), __file__)
    output_filename = os.path.join(os.getcwd(), filename)
    data = data.to_csv(output_filename)
    new_file = File(path = output_filename, parentId = data_parent_id)
    new_file = syn.store(new_file)
    os.remove(output_filename)
    
    syn.setProvenance(new_file, 
                      activity = Activity(used = synId, 
                                          executed = get_script_id(syn, __file__, script_parent_id)))

## run main function ##
if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
    