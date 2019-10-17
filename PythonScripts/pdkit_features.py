import sys
import os
import pandas as pd
import numpy as np
import pdkit
from pdkit.gait_time_series import GaitTimeSeries
from pdkit.gait_processor import GaitProcessor
from myutils import get_acceleration_ts, normalize_feature
import ast

"""
Applicable to walking motions
"""
def pdkit_pipeline(filepath, var):
    ### Process data to be usable by pdkit ###
    print(filepath)
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
    feature_dict["no_of_steps"] = no_of_steps
    feature_dict["mean_freeze_index"] = np.mean(freeze_index)
    feature_dict["median_freeze_index"] = np.median(freeze_index)
    feature_dict["max_freeze_index"] = np.max(freeze_index)
    feature_dict["count_freeze_index"] = freeze_count
    feature_dict["speed_of_gait"] = speed_of_gait
    feature_dict["gait_step_regularity"] = gait_step_regularity
    feature_dict["gait_stride_regularity"] = gait_stride_regularity
    feature_dict["gait_symmetry"] = gait_symmetry
    feature_dict["frequency_of_peaks"] = frequency_of_peaks
        
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
            print(pathfile)
            data[pathfile[:-8] + "features_{}".format(coord)] = data[pathfile].apply(pdkit_pipeline, var = coord)
    return data
def pdkit_normalize(data):
    for feature in [feat for feat in data.columns if "features" in feat]:
        print(feature)
        data = normalize_feature(data, feature)
    return data