import sys
import warnings
import pandas as pd
import numpy as np
import synapseclient as sc
warnings.simplefilter("ignore")

## login to synapse
syn = sc.login()

## demographic data ## 

if version == "V1":
    demographic_data = syn.tableQuery("SELECT distinct(healthCode) as healthCode, \
                                        age, education, gender   \
                                        diagnosis as PD from syn7222419")
                        .asDataFrame()) \
                        .drop_duplicates("healthCode", keep = "first") \
                        .reset_index(drop = True)) 
else:
    demographic_data = syn.tableQuery("SELECT distinct(healthCode) as healthCode, \
                                        birthYear, education, sex   \
                                        diagnosis as PD from syn7222419")

    
    
    
    demographic_data = demographic_data[demographic_mpV2_data["PD"] != "no_answer"] 
    demographic_data["PD"] = demographic_mpV2_data["PD"].map({"parkinsons":1, 
                                                               "control":0})
else:
    