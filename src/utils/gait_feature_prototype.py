import sys
import pdkit
import synapseclient as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import query_utils as query
import new_gait_feature_utils as gproc
from sklearn import metrics
import time
warnings.simplefilter("ignore")



def compute_gait_feature_per_window(data, orientation):
    """
    A modified function to calculate feature per 5 seconds window chunk
    parameter: 
    `filepath`    : time-series (filepath)
    `orientation` : coordinate orientation of the time series
    returns number of steps per chunk based on each recordIds
    """
    ts = data.copy()
    window_size = 512
    step_size = 100
    jPos = window_size + 1
    i = 0
    ts_arr = []
    while jPos < len(ts):
        time = jPos
        jStart = jPos - window_size
        subset = ts.iloc[jStart:jPos]
        window_duration = subset.td[-1] - subset.td[0]
        sample_rate = subset.shape[0]/window_duration
        print(sample_rate)
        gp = pdkit.GaitProcessor(duration = window_duration,
                                    cutoff_frequency = 5,
                                    filter_order = 4,
                                    delta = 0.5, 
                                    sampling_frequency = 100)
        var = subset[orientation].var()
        try:
            if (var) < 1e-2:
                heel_strikes = 0
            else:
                heel_strikes = len(gp.heel_strikes(subset[orientation])[1])
        except:
            heel_strikes = 0  
        try:
            gait_step_regularity = gp.gait_regularity_symmetry(subset[orientation])[0]
        except:
            gait_step_regularity = 0
        try:
            gait_stride_regularity = gp.gait_regularity_symmetry(subset[orientation])[1]
        except:
            gait_stride_regularity = 0
        try:
            gait_symmetry = gp.gait_regularity_symmetry(subset[orientation])[2]
        except:
            gait_symmetry = 0
        try:
            frequency_of_peaks = gp.frequency_of_peaks(subset[orientation])
        except:
            frequency_of_peaks = 0
        try:
            freeze_index_arr = gp.freeze_of_gait(subset[orientation])[1]
            min_freeze_index = np.min(freeze_index_arr)
            max_freeze_index = np.max(freeze_index_arr)
            mean_freeze_index = np.mean(freeze_index_arr)
            freeze_occurences = \
                (sum(i > 2.5 for i in gp.freeze_of_gait(subset[orientation])[1]))
        except:
            min_freeze_index = 0
            max_freeze_index = 0
            mean_freeze_index = 0
            freeze_occurences = 0
        try:
            speed_of_gait = gp.speed_of_gait(subset[orientation], wavelet_level = 6)
        except:
            speed_of_gait = 0
        jPos = jPos + step_size
        i = i + 1

    ## on each time-window chunk collect data into numpy array
        ts_arr.append({"time": time, 
                    "steps": heel_strikes, 
                    "cadence": heel_strikes/(window_size/100),
                    "variance": var,
                    "gait_step_regularity":gait_step_regularity,
                    "gait_stride_regularity":gait_stride_regularity,
                    "gait_symmetry":gait_symmetry,
                    "frequency_of_peaks":frequency_of_peaks,
                    "max_energy_freeze_index":max_freeze_index,
                    "mean_energy_freeze_index":mean_freeze_index,
                    "min_energy_freeze_index":min_freeze_index,
                    "freeze_occurences": freeze_occurences,
                    "freeze_occurences_per_sec": freeze_occurences/(window_size/100),
                    "speed_of_gait": speed_of_gait,
                    "window_duration": window_duration,                    
                    "window_end": subset.td[-1],
                    "window_start": subset.td[0]})
    return ts_arr


def subset_data_non_zero_runs(data, zero_runs_cutoff):
    """
    Function to subset data from zero runs heel strikes 
    that exceeded the cutoff threshold (consecutive zeros)
    parameter:
        `data`             : dataframe containing columns of chunk time and heel strikes per chunk
        `zero_runs_cutoff` : threshold of how many consecutive row of zeros that will be remove from the dataframe
    returns a subset of non-zero runs pd.DataFrame
    """
    z_runs_threshold = []
    for value in zero_runs(data["steps"]):
        # if not moving by this duration (5 seconds)
        if (value[1]-value[0]) >= zero_runs_cutoff:
            z_runs_threshold.append(value)
    z_runs_threshold = np.array(z_runs_threshold)
    for i in z_runs_threshold:
        data = data.loc[~data.index.isin(range(i[0],i[1]))]
    return data.reset_index(drop = True)

def zero_runs(array):
    """
    Function to search zero runs in an np.array
    parameter:
        `array`  : np array
    returns N x 2 np array matrix containing zero runs
    format of returned data: np.array([start index of zero occurence, end index of zero occurence], ...) 
    """
    # Create an array that is 1 where a is 0, and pad each end with an extra 0.
    iszero = np.concatenate(([0], np.equal(array, 0).view(np.int8), [0]))
    absdiff = np.abs(np.diff(iszero))
    # Runs start and end where absdiff is 1.
    ranges = np.where(absdiff == 1)[0].reshape(-1, 2)
    return ranges


def gait_processor_pipeline(filepath, orientation):
    """
    Function of data pipeline for subsetting data from rotational movements, retrieving rotational features, 
    removing low-variance longitudinal data and PDKIT estimation of heel strikes based on 2.5 secs window chunks
    parameters:
    
        `data`: string of pathfile, or pandas dataframe
        `orientation`: orientation of featurized data
    
    returns a featurized dataframe of rotational features and number of steps per window sizes
    """    
    accel_ts    = query.get_sensor_ts_from_filepath(filepath = filepath, 
                                        sensor = "userAcceleration")
    rotation_ts = query.get_sensor_ts_from_filepath(filepath = filepath, 
                                        sensor = "rotationRate")
    
    # return errors # 
    if not isinstance(accel_ts, pd.DataFrame):
        return "#ERROR"
    
    if not isinstance(rotation_ts, pd.DataFrame):
        return "#ERROR"

    acceleration_fs = accel_ts.shape[0]/accel_ts["td"][-1]
    rotation_fs = rotation_ts.shape[0]/rotation_ts["td"][-1]
    rotation_ts = calculate_rotation(rotation_ts, "y")
    rotation_occurences = rotation_ts[rotation_ts["aucXt"] > 2]
    data = gproc.create_overlay_data(accel_ts, rotation_ts)
    data = data.reset_index()
    walking_seqs = gproc.separate_array_sequence(np.where(data["aucXt"]<2)[0])
    gait_feature_arr = []
    for seqs in walking_seqs:
        data_seqs = data.loc[seqs[0]:seqs[-1]].set_index("time")
        gait_feature_arr.append(calculate_number_of_steps_per_window(data = data_seqs, 
                                                                     orientation = orientation))
    return [j for i in gait_feature_arr for j in i]

def calculate_rotation(data, orientation):
    """
    function to calculate rotational movement gyroscope AUC * period of zero crossing
    parameter:
    `data`  : pandas dataframe
    `orient`: orientation (string)
    returns dataframe of calculation of auc and aucXt
    """
    start = 0
    dict_list = {}
    dict_list["td"] = []
    dict_list["auc"] = []
    dict_list["turn_duration"] = []
    dict_list["aucXt"] = []
    data[orientation] = gproc.butter_lowpass_filter(data = data[orientation], 
                                            sample_rate = 100, 
                                            cutoff=2, 
                                            order=2)
    zcr_list = gproc.detect_zero_crossing(data[orientation].values)
    for i in zcr_list: 
        x = data["td"].iloc[start:i+1].values
        y = data[orientation].iloc[start:i+1].values
        turn_duration = data["td"].iloc[i+1] - data["td"].iloc[start]
        start  = i + 1
        if (len(y) >= 2):
            auc   = np.abs(metrics.auc(x,y)) 
            aucXt = auc * turn_duration
            dict_list["td"].append(x[-1])
            dict_list["turn_duration"].append(turn_duration)
            dict_list["auc"].append(auc)
            dict_list["aucXt"].append(aucXt)
    data = pd.DataFrame(dict_list)
    return data

def featurize_wrapper(data):
    """
    wrapper function for multiprocessing jobs
    parameter:
    `data`: takes in pd.DataFrame
    returns a json file featurized data
    """
    data["walk_features"] = data["walk_motion.json_pathfile"].apply(gait_processor_pipeline, orientation = "y")
    return data



def main():
    syn = sc.login()
    matched_demographic = query.get_file_entity(syn, "syn21482502")

    ## healthcode from version 1 ##
    hc_arr_v1 = (matched_demographic["healthCode"][matched_demographic["version"] == "mpower_v1"].unique())
    query_data_v1 = query.get_walking_synapse_table(syn, 
                                                    "syn10308918", 
                                                    "MPOWER_V1", 
                                                    healthCodes = hc_arr_v1)
    data_return   = query_data_v1[[feature for feature in query_data_v1.columns if "outbound" not in feature]]
    data_outbound = query_data_v1[[feature for feature in query_data_v1.columns if "return" not in feature]]
    query_data_v1 = pd.concat([data_outbound, data_return])## combine return and outbound                   
    arr_outbound = query_data_v1["deviceMotion_walking_outbound.json.items_pathfile"].dropna()
    arr_return = query_data_v1["deviceMotion_walking_return.json.items_pathfile"].dropna()
    query_data_v1["walk_motion.json_pathfile"] = pd.concat([arr_outbound, arr_return])

    ## healthcode from version 2 ## 
    hc_arr_v2 = (matched_demographic["healthCode"][matched_demographic["version"] == "mpower_v2"].unique())
    query_data_v2 = query.get_walking_synapse_table(syn, 
                                                    "syn12514611", 
                                                    "MPOWER_V2", 
                                                    healthCodes = hc_arr_v2)
    path_data = pd.concat([query_data_v1, query_data_v2]).reset_index(drop = True)                                             
    path_data = query.parallel_func_apply(path_data, featurize_wrapper, 16, 250)
    query.save_data_to_synapse(syn = syn, data = data, 
                            output_filename = "new_gait_features_matched2.csv",
                            data_parent_id = "syn20816722")

if __name__ ==  '__main__': 
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))