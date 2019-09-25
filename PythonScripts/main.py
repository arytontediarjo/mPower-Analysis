import os
import pandas as pd
import numpy as np
import synapseclient as sc
from synapseclient import Entity, Project, Folder, File, Link
from multiprocessing import Pool
import time
import warnings
from utils import getSynapseData
from pdkit_features import pdkitFeaturize
from spectral_flatness import sfm_auc
from pandarallel import pandarallel
warnings.simplefilter("ignore")


def parallelize_dataframe(df, func):
    
    ### split dataframe into 250 partitions ###
    df_split = np.array_split(df, 250)
    ### instantiate 16 processors as EC2 instance has 8 cores ###
    pool = Pool(16)
    ### map function into each pools ###
    map_values = pool.map(func, df_split)
    ### concatenate dataframe into one ###
    df = pd.concat(map_values)
    ### close pools
    pool.close()
    pool.join()
    return df


def featurize(data):
    
    for coord in ["x", "y", "z", "AA"]:
        
        ##### uncomment for pdkit
        # data["accel_outbound_features_{}".format(coord)] = data["accel_outbound_pathfile"].apply(pdkitFeaturize, var = coord)
        # data["accel_return_features_{}".format(coord)] = data["accel_return_pathfile"].apply(pdkitFeaturize,  var = coord)
        # data["accel_resting_features_{}".format(coord)] = data["accel_rest_pathfile"].apply(pdkitFeaturize,  var = coord)
        # data["userAccel_outbound_features_{}".format(coord)] = data["deviceMotion_outbound_pathfile"].apply(pdkitFeaturize, var = coord)
        # data["userAccel_return_features_{}".format(coord)] = data["deviceMotion_return_pathfile"].apply(pdkitFeaturize, var = coord)
        # data["userAccel_resting_features_{}".format(coord)] = data["deviceMotion_rest_pathfile"].apply(pdkitFeaturize, var = coord)
        data["sfm_auc_{}".format(coord)] =  data["deviceMotion_rest_pathfile"].parallel_apply(sfm_auc, var = coord)
    return data

def getHealthCodeInfo(syn, synid):
    entity = syn.get(synid)
    df = pd.read_csv(entity["path"], sep = "\t")
    healthcode_subset = "({})".format([i for i in df["healthCode"]]).replace("[", "").replace("]", "")
    filtered_healthcode_df = df[["healthCode", "age", "gender", "PD", "education", "Enter_State", "UTC_offset"]]
    return healthcode_subset, filtered_healthcode_df

def main():
    
    ## login
    syn = sc.login()
    pandarallel.initialize()
    
    ## get demographic information
    healthcode_subset, filtered_healthcode_df = getHealthCodeInfo(syn, "syn8381056")
    
    ## process data ##
    data = getSynapseData(syn, healthcode_subset)
    data = parallelize_dataframe(data, featurize)
    data = pd.merge(data, filtered_healthcode_df, how = "left", on = "healthCode")
    data = data[[feat for feat in data.columns if "path" not in feat]]
    
    
    ## save data to synapse ##
    data.to_csv("~/spectral_flatness_v2.csv")
    new_file = File(path = "~/spectral_flatness_v2.csv", parentId = "syn20816722")
    new_file = syn.store(new_file)
    
if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
