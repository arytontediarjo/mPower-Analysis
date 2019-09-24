import sys
import os
import pandas as pd
import numpy as np
import pdkit
from pdkit.gait_time_series import GaitTimeSeries
from pdkit.gait_processor import GaitProcessor
from utils import processAcceleration
import ast


def pdkitFeaturize(data, var):
    
    ### if filepath is empty ###
    if data == "EMPTY FILEPATHS":
        return data
    
    ### Process data to be usable by pdkit ###
    data = processAcceleration(data)
    
    
    
    ### if filepaths are not empty but accelerometer data is empty ###
    if data.shape[0] == 0:
        return "NO ACCELEROMETER DATA"
    
    ### parse through gait processor to retrieve resampled signal
    gp = pdkit.GaitProcessor(duration=data.td[-1])
    data = gp.resample_signal(data)
    
    ### instantiate empty dictionary ###
    feature_dict = {}
    try:  
        no_of_steps = gp.gait(data[var])[0]
        gait_step_regularity, gait_stride_regularity, gait_symmetry = gp.gait_regularity_symmetry(data[var])
        frequency_of_peaks = gp.frequency_of_peaks(data[var])
        freeze_time, freeze_index, locomotor_freeze_index = gp.freeze_of_gait(data[var])
        freeze_count = sum(i > 2.0 for i in freeze_index)
        speed_of_gait = gp.speed_of_gait(data[var], wavelet_level = 6)
    
    ### except function, happens when there is no movement, resting features
    except:
        no_of_steps = 0
        gait_step_regularity = 0
        gait_stride_regularity = 0
        gait_symmetry = 0
        frequency_of_peaks = 0
        freeze_time = 0
        freeze_index = 0
        locomotor_freeze_index = 0
        freeze_count = 0
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