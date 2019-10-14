import sys
import os
import pandas as pd
import numpy as np
import pdkit
from pdkit.gait_time_series import GaitTimeSeries
from pdkit.gait_processor import GaitProcessor
from myutils import get_acceleration_ts
import ast

"""
Applicable to walking motions
"""
def pdkit_pipeline(filepath, var):
    ### Process data to be usable by pdkit ###
    data = get_acceleration_ts(filepath)
    ### parse through gait processor to retrieve resampled signal
    try:
        ### if filepath is empty or have no accelerometer data ###
        if isinstance(data, (str, type(None))):
            return data
        gp = pdkit.GaitProcessor(duration=data.td[-1])
    except IndexError:
        return data
    
    data = gp.resample_signal(data)
    ### instantiate empty dictionary ###
    feature_dict = {}
    try:  
        no_of_steps = gp.gait(data[var])[0]
    except:
        no_of_steps = 0
    try:
        gait_step_regularity, gait_stride_regularity, gait_symmetry = gp.gait_regularity_symmetry(data[var])
    except:
        gait_step_regularity = 0
        gait_stride_regularity = 0
        gait_symmetry = 0
    try:
        frequency_of_peaks = gp.frequency_of_peaks(data[var])
    except:
        frequency_of_peaks = 0
    try:
        freeze_index = gp.freeze_of_gait(data[var])[1]
    except:
        freeze_index = 0
    try:
        freeze_count = sum(i > 2.0 for i in freeze_index)
    except:
        freeze_count = 0
    try:
        speed_of_gait = gp.speed_of_gait(data[var], wavelet_level = 6)
    except:
        speed_of_gait = 0
        
    ### fill in values to each keys
    feature_dict["no_of_steps_".format(var)] = no_of_steps
    feature_dict["mean_freeze_index_".format(var)] = np.mean(freeze_index)
    feature_dict["median_freeze_index_".format(var)] = np.median(freeze_index)
    feature_dict["max_freeze_index_".format(var)] = np.max(freeze_index)
    feature_dict["count_freeze_index_".format(var)] = freeze_count
    feature_dict["speed_of_gait_".format(var)] = speed_of_gait
    feature_dict["gait_step_regularity_".format(var)] = gait_step_regularity
    feature_dict["gait_stride_regularity_".format(var)] = gait_stride_regularity
    feature_dict["gait_symmetry_".format(var)] = gait_symmetry
    feature_dict["frequency_of_peaks_".format(var)] = frequency_of_peaks
        
    ## Final Check to clear nan values to zero ##
    for k, v in feature_dict.items():
        if np.isnan(v):
            feature_dict[k] = 0
        
    return feature_dict

def pdkit_featurize(data):
    for coord in ["x", "y", "z", "AA"]:
        for pathfile in [_ for _ in data.columns if ("pathfile" in _) 
                                                    and ("pedometer" not in _)
                                                    and ("balance" not in _)
                                                    and ("rest" not in _)]:
            data[pathfile[:-8] + "_features_{}".format(coord)] = data[pathfile].apply(pdkit_pipeline, var = coord)
    return data.fillna("#ERROR")

def pdkit_normalize(data):
    for i in [feat for feat in data.columns if "features" in feat]:
        data["no_of_steps {}".format(i)] = data[i].apply(normalize_dict, 
                                                         key = "no_of_steps_")
        data["median_freeze_index {}".format(i)] = data[i].apply(normalize_dict, 
                                                                 key = "median_freeze_index_")
        data["count_freeze_index {}".format(i)] = data[i].apply(normalize_dict, 
                                                                key = "count_freeze_index_")
        data["speed_of_gait {}".format(i)] = data[i].apply(normalize_dict, 
                                                           key = "speed_of_gait_")
        data["gait_step_regularity {}".format(i)] = data[i].apply(normalize_dict, 
                                                                  key = "gait_step_regularity_")
        data["gait_stride_regularity {}".format(i)] = data[i].apply(normalize_dict, 
                                                                    key = "gait_stride_regularity_")
        data["gait_symmetry {}".format(i)] = data[i].apply(normalize_dict, 
                                                           key = "gait_symmetry_")
        data["frequency_of_peaks {}".format(i)] = data[i].apply(normalize_dict, 
                                                                key = "frequency_of_peaks_")
        data = data.drop([i], axis = 1)
    return data

"""
Function to normalize dictionary on each column of PDKIT features
"""
def normalize_dict(params, key):
    try:
        dict_ = ast.literal_eval(params)
    except:
        return np.NaN
    return dict_[key]