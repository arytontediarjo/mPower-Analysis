import sys
import os
import warnings
import time 
import pandas as pd
import numpy as np
import utils as utils
import preprocessing_utils as proc
warnings.simplefilter("ignore")


WALKING_V1 = "syn21046180"
WALKING_V2 = "syn21046181"
WALKING_PASSIVE = "syn21046184"
DATA_PARENT_ID = "syn21118104"
GIT_URL = "https://github.com/arytontediarjo/mPower-Analysis/blob/master/PythonScripts/create_model_data.py"




def main():

    ### retrieve version 1 data ###
    dataV1 = utils.get_file_entity(WALKING_V1)
    dataV1 = proc.preprocess(dataV1, "max", True)
    dataV1["version"] = "V1"


    ### retrieve data from mPower version 2 ###
    ### side note: some data are prune to match the distribution to those in V1 ###
    dataV2 = utils.get_file_entity(WALKING_V2)
    ### run data into preprocessing sklearn class to collapse each features by its maximum ###
    dataV2              = proc.preprocess(dataV2, "max", True)
    ### annotate each data versions ###
    dataV2["version"]   = "V2"


    ### retrieve passive data ####
    dataPassive         = utils.get_file_entity(WALKING_PASSIVE)
    ### Ensure that per healthcodes is not only contributing one data (>5 recordIds threshold) ###
    dataPassive         = proc.preprocess(dataPassive, "max", True)
    ## set bottom limit of age that is not lower than the lowest age at version 1 to sustain distribution ##
    dataPassive["version"] = "PASSIVE"


    for data, version, source_table_id in zip([dataV1, dataV2, dataPassive], 
                                                ["v1", "v2", "passive"],
                                                [WALKING_V1, WALKING_V2, WALKING_PASSIVE]):
    ### save script to synapse and save cleaned dataset to synapse ###
        utils.save_data_to_synapse( data             = data, 
                                    output_filename = "max_collapsed_walking_feature_%s.csv" %version, 
                                    data_parent_id  = DATA_PARENT_ID,
                                    used_script     = GIT_URL,
                                    source_table_id = source_table_id) 

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))