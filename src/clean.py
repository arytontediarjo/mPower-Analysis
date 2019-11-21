import pandas as pd
import synapseclient as sc
import time
import sys
from utils.munging_utils import get_file_entity, fix_column_name, save_data_to_synapse
from utils.preprocessing_utils import preprocess, addAdditionalFeatures_viz, collapseFeatures
from datetime import datetime
import numpy as np
import warnings
warnings.simplefilter("ignore")


### CONSTANTS ###
MPOWER_GAIT_DATA_V1 = "syn21111818"
MPOWER_DEMO_DATA_V1 = "syn10371840"
MPOWER_GAIT_DATA_V2 = "syn21113231"
MPOWER_DEMO_DATA_V2 = "syn15673379"
MPOWER_GAIT_DATA_PASSIVE = "syn21114136"
EMS_PROF_DATA = "syn10235463"
EMS_DEMO_DATA = "syn10295288"
EMS_GAIT_DATA = "syn21256442"
EMS_PASSIVE_DATA = "syn10651116"
METADATA_COLS  = ['recordId', 'healthCode', 'appVersion', 
                    'phoneInfo', 'createdOn', 'PD', 'MS',
                    'gender', 'age', 'version']
GIT_URL = "https://github.com/arytontediarjo/mPower-Analysis/blob/master/src/clean.py"

syn = sc.login()

def create_mPowerV1_data(GAIT_DATA, DEMO_DATA, output_filename):
    demo_data = syn.tableQuery("SELECT * FROM {} where dataGroups\
                               NOT LIKE '%test_user%'".format(DEMO_DATA)).asDataFrame()
    gait_data = get_file_entity(GAIT_DATA)
    demo_data = demo_data[["healthCode", "gender", "age",
                           "professional_diagnosis", "inferred_diagnosis"]].reset_index(drop = True)
    data = pd.merge(gait_data, demo_data, on = "healthCode", how = "inner")
    data_return   = data[[feature for feature in data.columns if "outbound" not in feature]]
    data_outbound = data[[feature for feature in data.columns if "return" not in feature]]
    data = pd.concat([fix_column_name(data_outbound), fix_column_name(data_return)])
    data = data.dropna(subset = ["inferred_diagnosis"], thresh = 1)
    data["PD"] = data["inferred_diagnosis"].map({True :1.0, False:0.0})
    data = data[(data["gender"] == "Female") | (data["gender"] == "Male")]
    data["age"] = data["age"].apply(lambda x: float(x))
    data = data[(data["age"] <= 100) & (data["age"] >= 0)]
    data["gender"] = data["gender"].apply(lambda x: x.lower())
    data = fix_column_name(data)
    data = data.reset_index(drop = True)
    data = data[[feat for feat in data.columns if ("." in feat) or (feat in METADATA_COLS)]]
    save_data_to_synapse(data = data, 
                         output_filename = output_filename,
                         data_parent_id = "syn21267355")
    return data


def create_mPowerV2_data(GAIT_DATA, DEMO_DATA, output_filename):
    demo_data = syn.tableQuery("SELECT birthYear, healthCode, diagnosis, sex FROM {} \
                                where dataGroups NOT LIKE '%test_user%'".format(DEMO_DATA)).asDataFrame()
    gait_data = get_file_entity(GAIT_DATA)
    data   = pd.merge(gait_data, demo_data, how = "inner", on = "healthCode")
    data   = data[data["diagnosis"] != "no_answer"] 
    data["PD"] = data["diagnosis"].map({"parkinsons":1, "control":0})
    data["age"] = data["birthYear"].apply(lambda year: datetime.now().year - year)
    data = data.rename({"sex":"gender"}, axis = 1)
    data = fix_column_name(data)
    data = data.reset_index(drop = True)
    data = data[[feat for feat in data.columns if ("." in feat) or (feat in METADATA_COLS)]]
    save_data_to_synapse(data = data, 
                        output_filename = output_filename,
                        data_parent_id = "syn21267355")
    return data

def create_elevateMS_data(GAIT_DATA, DEMO_DATA, output_filename):
    demo_data = syn.tableQuery("SELECT healthCode, dataGroups, 'demographics.gender', 'demographics.age' FROM {}\
                                    where dataGroups NOT LIKE '%test_user%'".format(DEMO_DATA)).asDataFrame()
    gait_data    = get_file_entity(GAIT_DATA)
    data         = pd.merge(gait_data, demo_data, how = "inner", on = "healthCode")
    data = data.dropna(subset = ["demographics.gender"])
    data["MS"] = data["dataGroups"].map({"ms_patient":1, "control":0})
    data  = data.rename({"demographics.gender" :"gender",
                         "demographics.age"    : "age"}, axis = 1)
    data = fix_column_name(data)
    data = data.reset_index(drop = True)
    data = data[[feat for feat in data.columns if ("." in feat) or (feat in METADATA_COLS)]]
    save_data_to_synapse(data = data, 
                        output_filename = output_filename,
                        data_parent_id = "syn21267355")
    return data

def combine_data(*dataframes):
    dataframe_list = []
    for data in dataframes:
        dataframe_list.append(data)
    data = pd.concat(dataframe_list).reset_index(drop = True)
    data = data[[feat for feat in data.columns if ("." in feat) or (feat in METADATA_COLS)]]
    data = data[(data != "#ERROR").all(axis = 1)]
    data["is_control"] = data.apply(lambda x: 0 if ((x["PD"] == 0) or (x["MS"] == 0)) else 1, axis = 1)
    data[[_ for _ in data.columns if "." in _]] = \
        data[[_ for _ in data.columns if "." in _]].apply(pd.to_numeric)
    data.drop(["y.duration", "z.duration", "AA.duration"], axis = 1, inplace = True) 
    data.rename({"x.duration": "duration"}, axis = 1, inplace = True)
    save_data_to_synapse(data = data.reset_index(drop = True), 
                        output_filename = "combined_gait_data.csv",
                        data_parent_id  = "syn21267355",
                        source_table_id = ["syn21256442", "syn21114136", "syn21111818", "syn21113231"],
                        used_script = GIT_URL)

def main():
    dataV1                    = create_mPowerV1_data(GAIT_DATA = MPOWER_GAIT_DATA_V1, 
                                                     DEMO_DATA = MPOWER_DEMO_DATA_V1, 
                                                     output_filename = "pdkit_mpowerv1_active_full.csv")
    dataV1["version"]         = "V1"
    dataV2                    = create_mPowerV2_data(GAIT_DATA = MPOWER_GAIT_DATA_V2, 
                                                     DEMO_DATA = MPOWER_DEMO_DATA_V2, 
                                                     output_filename = "pdkit_mpowerv2_active_full.csv")
    dataV2["version"]         = "V2"
    dataPassive               = create_mPowerV2_data(GAIT_DATA = MPOWER_GAIT_DATA_PASSIVE, 
                                                     DEMO_DATA = MPOWER_DEMO_DATA_V2, 
                                                     output_filename = "pdkit_mpowerv2_passive_full.csv")
    dataPassive["version"]    = "PD_passive"
    dataEMS_active            = create_elevateMS_data(GAIT_DATA = EMS_GAIT_DATA, 
                                                    DEMO_DATA = EMS_PROF_DATA, 
                                                    output_filename = "pdkit_ems_active_full.csv")
    dataEMS_active["version"] = "MS_active"
    combine_data(dataV1, dataV2, dataPassive, dataEMS_active)

"""
Run main function and record the time of script runtime
"""
if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
