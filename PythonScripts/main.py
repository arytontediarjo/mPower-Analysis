import os
import pandas as pd
import numpy as np
import synapseclient as sc
from synapseclient import Entity, Project, Folder, File, Link
import multiprocessing as mp
from multiprocessing import Pool
import time
import warnings
from myutils import get_synapse_table
from pdkit_features import pdkit_featurize, pdkit_normalize
from spectral_flatness import sfm_featurize
import argparse
warnings.simplefilter("ignore")


## Read Arguments
def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", default= "data.csv",
                        help = "Path for output results")
    parser.add_argument("--num-cores", default= 16,
                        help = "Number of Cores, negative number not allowed")
    parser.add_argument("--num-chunks", default= 10,
                        help = "Number of sample per partition, no negative number")
    parser.add_argument("--featurize", default= "spectral-flatness",
                        help = "Features choices: 'pdkit' or 'spectral-flatness' ")
    parser.add_argument("--table-id", default= "syn7222425", ## mpower V1
                        help = "mpower gait table to query from")
    parser.add_argument("--filtered", action='store_true', 
                        help = "filter healthcodes")
    args = parser.parse_args()
    return args

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

def filter_healthcodes(syn, synId, is_filtered):
    ## get demographic information
    
    if is_filtered:
        filtered_entity = syn.get("syn8381056")
        filtered_healthcode_data = pd.read_csv(filtered_entity["path"], sep = "\t")
        filtered_healthcode_list = list(filtered_healthcode_data["healthCode"])
        return filtered_healthcode_list
    else:
        
        all_healthcode_list = list(syn.tableQuery("select distinct(healthCode) as healthCode from {}".format(synId))
                                   .asDataFrame()["healthCode"])
        print(all_healthcode_list)
        return all_healthcode_list
        
        


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
    synId = args.table_id
    is_filtered = args.filtered
    print(is_filtered)
    
    ## login
    syn = sc.login()
    
    ## process data ##
    data = get_synapse_table(syn, filter_healthcodes(syn, synId, is_filtered), synId)
    print(data)
        
    ## condition on choosing which features
    print("Retrieving {} Features".format(features))
    
    
    if features == "spectral-flatness":
        data = _parallelize_dataframe(data, sfm_featurize, cores, chunksize)
    elif features == "pdkit":
        data = _parallelize_dataframe(data, pdkit_featurize, cores, chunksize)
    print("parallelization process finished")
    data = data[[feat for feat in data.columns if "path" not in feat]]
    
    ## save data to local directory then to synapse ##
    file_path = "~/{}".format(filename)
    data.to_csv(file_path)
    new_file = File(path = file_path, parentId = "syn20816722")
    new_file = syn.store(new_file)
    
    
## Run Main Function
if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
