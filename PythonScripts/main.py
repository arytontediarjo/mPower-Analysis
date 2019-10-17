"""
Errors that will be catched:
Empty filepaths, Empty Features, Empty Files
Will be annotated as #ERROR universally
#ERROR will be removed on feature engineering
"""


import os
import pandas as pd
import numpy as np
import synapseclient as sc
from synapseclient import Entity, Project, Folder, File, Link, Activity
import multiprocessing as mp
from multiprocessing import Pool
import time
import warnings
from myutils import get_synapse_table, get_healthcodes, get_script_id
from pdkit_features import pdkit_featurize, pdkit_normalize
from spectral_flatness import sfm_featurize
import argparse
warnings.simplefilter("ignore")


"""
Function for parsing in argument given by client
"""
def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", default= "data.csv",
                        help = "Path for output results")
    parser.add_argument("--num-cores", default= 8,
                        help = "Number of Cores, negative number not allowed")
    parser.add_argument("--num-chunks", default= 10,
                        help = "Number of sample per partition, no negative number")
    parser.add_argument("--featurize", default= "spectral-flatness",
                        help = "Features choices: 'pdkit' or 'spectral-flatness' ")
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

"""
Main function to query mpower V1 Data 
Will be updated with mpower V2 Data
""" 
def main():
    ## Retrieve Arguments
    args = read_args()
    filename = args.filename ## name of the file
    cores = int(args.num_cores) ## number of cores
    chunksize = int(args.num_chunks) ## number of chunks
    features = args.featurize ## which features to query
    synId = args.table_id ## which table to query from
    is_filtered = args.filtered ## filter the dataset
    script_parent_id = args.script_parent_id
    data_parent_id = args.data_parent_id
    
    ## login
    syn = sc.login()
    ## process data ##
    data = get_synapse_table(syn, get_healthcodes(syn, synId, is_filtered), synId)
    
    ## condition on choosing which features
    print("Retrieving {} Features".format(features))
    if features == "spectral-flatness":
        data = _parallelize_dataframe(data, sfm_featurize, cores, chunksize)
    elif features == "pdkit":
        data = _parallelize_dataframe(data, pdkit_featurize, cores, chunksize)
        data = pdkit_normalize(data)
    print("parallelization process finished")
    data = data[[feat for feat in data.columns if "path" not in feat]]
    
    
    path_to_script = os.path.join(os.getcwd(), __file__)
    output_filename = os.path.join(os.getcwd(), filename)
    data = data.to_csv(output_filename)
    new_file = File(path = output_filename, parentId = data_parent_id)
    new_file = syn.store(new_file)
    os.remove(output_filename)
    
    syn.setProvenance(new_file, 
                      activity = Activity(used = synId, 
                                          executed = get_script_id(syn, __file__, "syn20987850")))
    
## Run Main Function ##
if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
