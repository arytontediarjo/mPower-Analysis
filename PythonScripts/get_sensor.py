import sys
import os
import warnings
import pandas as pd
import numpy as np
import synapseclient as sc
import time
from myutils import get_sensor_types, get_units, get_synapse_table, 
                    store_to_synapse, get_script_id, get_healthcodes
import argparse
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
    return data

def main(): 
    args = read_args()
    filename = args.filename ## name of the file
    synId = args.table_id ## which table to query from
    is_filtered = args.filtered ## filter the dataset?
    
    syn = sc.login()
    data = get_synapse_table(syn, get_healthcodes(syn, synId, is_filtered), synId)
    data = _parallelize_dataframe(data, _featurize, 16, 250)
    
    ## store data and script ##
    path_to_script = os.path.join(os.getcwd(), __file__)
    output_filename = os.path.join(os.getcwd(), filename)
    store_script = store_to_synapse(syn  = syn,    filename = path_to_script,
                                    data = np.NaN, parentId = "syn20987850")
    store_data = store_to_synapse(syn  = syn, filename  = output_filename,
                                  data = data, parentId = "syn20988708",
                                  source_id = "syn12514611", name = "sensor preprocessing",
                                  script_id = get_script_id(syn, __file__, "syn20987850"))
    
if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
    