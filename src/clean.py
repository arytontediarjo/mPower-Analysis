import pandas as pd
import synapseclient as sc
import time
import sys
from utils.query_utils import get_file_entity, fix_column_name, save_data_to_synapse
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
METADATA_COLS  = ['recordId', 'healthCode', 'appVersion', 
                    'phoneInfo', 'createdOn', 'PD', 'MS',
                    'gender', 'age', 'version']
GIT_URL = "https://github.com/arytontediarjo/mPower-Analysis/blob/master/src/clean.py"

syn = sc.login()

def create_mPowerV1_interim_gait_data(GAIT_DATA, DEMO_DATA):
    """
    Function to format mpower version 1 data,
    list of formatting done:
        -> Clean table from test users
        -> Combine raw data with demographic table
        -> Fix column naming convention
        -> Map diagnosis to binary values
        -> Clean data that is below the range of 0-100
        -> Filter gender to male and female
    Parameters:
    GAIT_DATA = Takes in raw featurized gait data on version 1 (synapse file entity)
    DEMO_DATA = Takes in demographic data (synapse table entity)
    returns a formatized dataset of featurized gait data with its respective demographic data
    """
    demo_data = syn.tableQuery("SELECT age, healthCode, inferred_diagnosis as PD, gender FROM {} where dataGroups\
                               NOT LIKE '%test_user%'".format(DEMO_DATA)).asDataFrame()
    demo_data = demo_data[(demo_data["gender"] == "Female") | (demo_data["gender"] == "Male")]
    demo_data = demo_data.dropna(subset = ["PD"], thresh = 1)                     ## drop if no diagnosis
    demo_data["PD"] = demo_data["PD"].map({True :1.0, False:0.0})                 ## encode as numeric binary
    demo_data["age"] = demo_data["age"].apply(lambda x: float(x))                 ## convert age to float
    demo_data = demo_data[(demo_data["age"] <= 100) & (demo_data["age"] >= 10)]   ## subset to realistic age ranges
    demo_data["gender"] = demo_data["gender"].apply(lambda x: x.lower())          ## lowercase gender for consistencies
    gait_data = get_file_entity(syn = syn, synid = GAIT_DATA)
    data = pd.merge(gait_data, demo_data, on = "healthCode", how = "inner")
    data_return   = data[[feature for feature in data.columns if "outbound" not in feature]]
    data_outbound = data[[feature for feature in data.columns if "return" not in feature]]
    data = pd.concat([fix_column_name(data_outbound), fix_column_name(data_return)])## combine return and outbound                                                   
    data = data.reset_index(drop = True)
    data = data[[feat for feat in data.columns if ("." in feat) or (feat in METADATA_COLS)]]
    return data


def create_mPowerV2_interim_gait_data(GAIT_DATA, DEMO_DATA):
    """
    Function to format mpower version 2 data,
    list of formatting done:
        -> Clean table from test users
        -> Combine raw data with demographic table
        -> Fix column naming convention
        -> Map diagnosis to binary values
        -> Clean data that is below the range of 0-100
        -> Filter gender to male and female
    Parameters:
    GAIT_DATA = Takes in raw featurized gait data on version 2(synapse file entity)
    DEMO_DATA = Takes in demographic data (synapse table entity)

    returns a formatized dataset of featurized gait data with its respective demographic data
    """
    demo_data = syn.tableQuery("SELECT birthYear, createdOn, healthCode, diagnosis as PD, sex as gender FROM {} \
                                where dataGroups NOT LIKE '%test_user%'".format(DEMO_DATA)).asDataFrame()
    demo_data        = demo_data[(demo_data["gender"] == "male") | (demo_data["gender"] == "female")]
    demo_data        = demo_data[demo_data["PD"] != "no_answer"]               
    demo_data["PD"]  = demo_data["PD"].map({"parkinsons":1, "control":0})
    demo_data["age"] = pd.to_datetime(demo_data["createdOn"], unit = "ms").dt.year - demo_data["birthYear"]
    demo_data = demo_data[(demo_data["age"] <= 100) & (demo_data["age"] >= 10)]
    demo_data = demo_data.drop(["birthYear", "createdOn"], axis = 1)                  
    gait_data = get_file_entity(syn = syn, synid = GAIT_DATA)
    data      = pd.merge(gait_data, demo_data, how = "inner", on = "healthCode")
    data      = fix_column_name(data)
    data      = data.reset_index(drop = True)
    data      = data[[feat for feat in data.columns if ("." in feat) or (feat in METADATA_COLS)]]
    return data

def create_elevateMS_interim_gait_data(GAIT_DATA, DEMO_DATA):
    """
    Function to format EMS data,
    list of formatting done:
        -> Clean table from test users
        -> Combine raw data with demographic table
        -> Fix column naming convention
        -> Map diagnosis to binary values
    Parameters:
    GAIT_DATA = Takes in raw featurized gait data on EMS (synapse file entity)
    DEMO_DATA = Takes in demographic data (synapse table entity)

    returns a formatized dataset of featurized gait data with its respective demographic data
    """
    demo_data = syn.tableQuery("SELECT healthCode, dataGroups as MS, 'demographics.gender' as gender,\
                            'demographics.age' as age FROM {} where dataGroups NOT LIKE '%test_user%'".format(DEMO_DATA)).asDataFrame()
    demo_data = demo_data[(demo_data["gender"] == "Male") | (demo_data["gender"] == "Female")]
    demo_data["gender"] = demo_data["gender"].apply(lambda x: x.lower())
    demo_data = demo_data[(demo_data["age"] <= 100) & (demo_data["age"] >= 10)]
    demo_data["MS"] = demo_data["MS"].map({"ms_patient":1, "control":0})
    gait_data    = get_file_entity(syn = syn, synid = GAIT_DATA)
    data         = pd.merge(gait_data, demo_data, how = "inner", on = "healthCode")
    data         = fix_column_name(data)
    data         = data.reset_index(drop = True)
    data         = data[[feat for feat in data.columns if ("." in feat) or (feat in METADATA_COLS)]]
    return data

def _annotate_classes(PD_status, MS_status, version):
    if (version == "mpower_v1"  and PD_status == 1):
        return "mpower_v1_case"
    elif (version == "mpower_v1" and PD_status == 0):
        return "mpower_v1_control"
    elif (version == "mpower_v2"  and PD_status == 1):
        return "mpower_v2_case"
    elif (version == "mpower_v2" and PD_status == 0):
        return "mpower_v2_control"
    elif (version == "mpower_passive" and PD_status == 1):
        return "mpower_passive_case"
    elif (version == "mpower_passive" and PD_status == 0):
        return "mpower_passive_control"
    elif (version == "ems_active"  and MS_status == 1):
        return "ems_case"
    else:
        return "ems_control"
    


def combine_gait_data(*dataframes):
    """
    Function to join all interim data into one readily used dataframe
    """
    dataframe_list = []
    for data in dataframes:
        dataframe_list.append(data)
    data = pd.concat(dataframe_list).reset_index(drop = True)
    data["PD"] = data["PD"].fillna(0)
    data["MS"] = data["MS"].fillna(0)
    data = data[(data != "#ERROR").all(axis = 1)]
    data["is_control"] = data.apply(lambda x: 1 if (x["PD"] == 0 and x["MS"] ==0) else 0, axis = 1)
    data["class"] = data.apply(lambda x: _annotate_classes(x["PD"], x["MS"], x["version"]), axis = 1)
    data[[_ for _ in data.columns if "." in _]] = data[[_ for _ in data.columns if "." in _]].apply(pd.to_numeric)
    data.drop(["y.duration", "z.duration", "AA.duration"], axis = 1, inplace = True) 
    data.rename({"x.duration": "duration"}, axis = 1, inplace = True)
    save_data_to_synapse(syn = syn,
                        data = data.reset_index(drop = True), 
                        output_filename = "combined_gait_data.csv",
                        data_parent_id  = "syn21267355",
                        source_table_id = [MPOWER_GAIT_DATA_V1, MPOWER_DEMO_DATA_V1, MPOWER_GAIT_DATA_V2, 
                                            MPOWER_DEMO_DATA_V2, MPOWER_GAIT_DATA_PASSIVE, EMS_PROF_DATA, EMS_GAIT_DATA],
                        used_script = GIT_URL)

"""
Main Function
"""
def main():
    dataV1                    = create_mPowerV1_interim_gait_data(GAIT_DATA = MPOWER_GAIT_DATA_V1, DEMO_DATA = MPOWER_DEMO_DATA_V1)
    dataV1["version"]         = "mpower_v1"
    dataV2                    = create_mPowerV2_interim_gait_data(GAIT_DATA = MPOWER_GAIT_DATA_V2, DEMO_DATA = MPOWER_DEMO_DATA_V2)
    dataV2["version"]         = "mpower_v2"
    dataPassive               = create_mPowerV2_interim_gait_data(GAIT_DATA = MPOWER_GAIT_DATA_PASSIVE, DEMO_DATA = MPOWER_DEMO_DATA_V2)
    dataPassive["version"]    = "mpower_passive"
    dataEMS_active            = create_elevateMS_interim_gait_data(GAIT_DATA = EMS_GAIT_DATA, DEMO_DATA = EMS_PROF_DATA)
    dataEMS_active["version"] = "ems_active"
    combine_gait_data(dataV1, dataV2, dataPassive, dataEMS_active)

"""
Run main function and record the time of script runtime
"""
if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))
