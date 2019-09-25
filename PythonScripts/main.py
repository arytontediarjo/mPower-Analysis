import os
import pandas as pd
import numpy as np
import synapseclient as sc
from synapseclient import Entity, Project, Folder, File, Link
import multiprocessing as mp
from multiprocessing import Pool
import time
import warnings
from utils import getSynapseData
from pdkit_features import pdkit_featurize
from spectral_flatness import sfm_featurize
import argparse
warnings.simplefilter("ignore")


## Read Arguments
def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--filename", default= "data.csv",
                        help = "Path for output results")
    parser.add_argument("--num-cores", default= mp.cpu_count(),
                        help = "Number of Cores, negative number not allowed")
    parser.add_argument("--num-chunks", default= 10,
                        help = "Number of sample per partition, no negative number")
    parser.add_argument("--featurize", default= "spectral-flatness",
                        help = "Features choices: 'pdkit' or 'spectral-flatness' ")
    args = parser.parse_args()
    return args

def parallelize_dataframe(df, func, no_of_processors):
    ### split dataframe into 250 partitions ###
    df_split = np.array_split(df, 250)
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

def getHealthCodeInfo(syn, synid):
    entity = syn.get(synid)
    df = pd.read_csv(entity["path"], sep = "\t")
    healthcode_subset = "({})".format([i for i in df["healthCode"]]).replace("[", "").replace("]", "")
    filtered_healthcode_df = df[["healthCode", "age", "gender", "PD", "education", "Enter_State", "UTC_offset"]]
    return healthcode_subset, filtered_healthcode_df

def main():
    ## Retrieve Arguments
    args = read_args()
    filename = args.filename
    cores = args.num_cores
    features = args.feature_choice
    
    ## login
    syn = sc.login()
    ## get demographic information
    healthcode_subset, filtered_healthcode_df = getHealthCodeInfo(syn, "syn8381056")
    ## process data ##
    data = getSynapseData(syn, healthcode_subset)
    
    
    ## condition on choosing which features
    print("Retrieving {} Features".format(features))
    if features == "spectral-flatness":
        data = parallelize_dataframe(data, sfm_featurize, cores)
    elif features == "pdkit":
        data = parallelize_dataframe(data, pdkit_featurize, features)
    
    data = pd.merge(data, filtered_healthcode_df, how = "left", on = "healthCode")
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
