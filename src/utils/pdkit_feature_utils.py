import sys
import os
import ast
import pandas as pd
import numpy as np
import pdkit
from pdkit.gait_time_series import GaitTimeSeries
from pdkit.gait_processor import GaitProcessor
from utils.query_utils import get_acceleration_ts, normalize_dict_features


def pdkit_pipeline(filepath, var):
    """
    Function to run pdkit package, it captures the duration of longitudinal data and the longitudinal data itself
    measure gait features like number of steps, symmetry measurement, regularity and freeze occurences
    parameter: filepath = filepath in .synapseCache
               var = which coordinate orientation to run pdkit pipeline on
    returns dictionary of pdkit gait features
    """
    ### Process data to be usable by pdkit ###
    data = get_sensor_ts(filepath = filepath, 
                                sensor = "userAcceleration")
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
    
    ### featurize pdkit features into the dictionary ###
    try:
        feature_dict["duration"]                = data.td[-1]
    except:
        feature_dict["duration"]                = 0
    try:  
        feature_dict["no_of_steps"]             = gp.gait(data[var])[0]
    except:
        feature_dict["no_of_steps"]             = 0    
    try:
        feature_dict["gait_step_regularity"]    = gp.gait_regularity_symmetry(data[var])[0]
    except:
        feature_dict["gait_step_regularity"]    = 0
    try:
        feature_dict["gait_stride_regularity"]  = gp.gait_regularity_symmetry(data[var])[1]
    except:
        feature_dict["gait_stride_regularity"]  = 0
    try:
        feature_dict["gait_symmetry"]           = gp.gait_regularity_symmetry(data[var])[2]
    except:
        feature_dict["gait_symmetry"]           = 0
    try:
        feature_dict["frequency_of_peaks"]      = gp.frequency_of_peaks(data[var])
    except:
        feature_dict["frequency_of_peaks"]      = 0
    try:
        feature_dict["max_freeze_index"]        = np.max(gp.freeze_of_gait(data[var])[1])
    except:
        feature_dict["max_freeze_index"]        = 0
    try:
        feature_dict["freeze_occurences"]       = sum(i > 2.0 for i in gp.freeze_of_gait(data[var])[1])
    except:
        feature_dict["freeze_occurences"]       = 0
    try:
        feature_dict["speed_of_gait"]           = gp.speed_of_gait(data[var], wavelet_level = 6)
    except:
        feature_dict["speed_of_gait"]           = 0
    return feature_dict


def pdkit_featurize(data):
    """
    Function to featurize the filepath data with pdkit features,
    loops through xyz and AA(resultant) signal coordinate orientation
    and run pdkit package on each data in given coordinate orientations
    
    parameter: filepath dataframe
    returns: featurized data
    """
    for coord in ["x", "y", "z", "AA"]:
        for feature in [_ for _ in data.columns if ("pathfile" in _) 
                                                    and ("balance" not in _)
                                                    and ("rest" not in _)
                                                    and ("coord" not in _)]:
            data["{}_features_{}".format(feature[:-8], coord)] = data[feature].apply(pdkit_pipeline, var = coord)
    return data


def normalize_pdkit_features(data):
    """
    Function to normalize dictionary into several key columns
    parameter: dataframe
    returns a normalized dataframe given a dictionary inside columns
    """
    for feature in [feat for feat in data.columns if "features" in feat]:
        data = normalize_dict_features(data, feature)
    return data