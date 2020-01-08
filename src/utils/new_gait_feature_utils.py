## package imports ##
import sys
import pdkit
import query_utils as query
import synapseclient as sc
import matplotlib.pyplot as plt
from scipy import signal
import warnings
from scipy.fftpack import (rfft, fftfreq)
from scipy.signal import (butter, lfilter, correlate, freqz)
import pandas as pd
import seaborn as sns
import numpy as np
from sklearn import metrics
from operator import itemgetter
import time
from itertools import *
warnings.simplefilter("ignore")


"""
TODO:
consolidate low pass filters parameter into classes
"""


## helper functions ##
def butter_lowpass_filter(data, sample_rate, cutoff=10, order=4, plot=False):
    """
        `Low-pass filter <http://stackoverflow.com/questions/25191620/
        creating-lowpass-filter-in-scipy-understanding-methods-and-units>`_ data by the [order]th order zero lag Butterworth filter
        whose cut frequency is set to [cutoff] Hz.
        :param data: time-series data,
        :type data: numpy array of floats
        :param: sample_rate: data sample rate
        :type sample_rate: integer
        :param cutoff: filter cutoff
        :type cutoff: float
        :param order: order
        :type order: integer
        :return y: low-pass-filtered data
        :rtype y: numpy array of floats
        :Examples:
        >>> from mhealthx.signals import butter_lowpass_filter
        >>> data = np.random.random(100)
        >>> sample_rate = 10
        >>> cutoff = 5
        >>> order = 4
        >>> y = butter_lowpass_filter(data, sample_rate, cutoff, order)
    """
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)

    if plot:
        w, h = freqz(b, a, worN=8000)
        plt.subplot(2, 1, 1)
        plt.plot(0.5*sample_rate*w/np.pi, np.abs(h), 'b')
        plt.plot(cutoff, 0.5*np.sqrt(2), 'ko')
        plt.axvline(cutoff, color='k')
        plt.xlim(0, 0.5*sample_rate)
        plt.title("Lowpass Filter Frequency Response")
        plt.xlabel('Frequency [Hz]')
        plt.grid()
        plt.show()
    y = lfilter(b, a, data)
    return y


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

def detect_zero_crossing(array):
    """
    Function to detect zero crossings in a time series signal data
    parameter:
        `array`: numpy array
    returns index location before sign change 
    """
    zero_crossings = np.where(np.diff(np.sign(array)))[0]
    return zero_crossings

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
    for value in zero_runs(data["heel_strikes"]):
        # if not moving by this duration (5 seconds)
        if (value[1]-value[0]) >= zero_runs_cutoff:
            z_runs_threshold.append(value)
    z_runs_threshold = np.array(z_runs_threshold)
    for i in z_runs_threshold:
        data = data.loc[~data.index.isin(range(i[0],i[1]))]
    return data.reset_index(drop = True)


def calculate_number_of_steps_per_window(data, orientation):
    """
    A modified function to calculate number of steps per 2.5 seconds window chunk
    parameter: 
    `filepath`    : time-series (filepath)
    `orientation` : coordinate orientation of the time series
    returns number of steps per chunk based on each recordIds
    """
    
    ts = data.copy()
    window_size = 256
    step_size = 50
    jPos = window_size + 1
    i = 0
    time = []
    variances = []
    
    sample_rate = ts.shape[0]/ts["td"][-1]
    ts["filtered_%s" %orientation] = butter_lowpass_filter(ts[orientation], 
                                                            sample_rate = 100, 
                                                            cutoff = 5, 
                                                            order = 4)
    time = []
    heel_strikes = []
    while jPos < len(ts):
        jStart = jPos - window_size
        time.append(jPos)
        subset = ts.iloc[jStart:jPos]
        gp = pdkit.GaitProcessor(duration = subset.td[-1] - subset.td[0],
                                    cutoff_frequency = 5,
                                    filter_order = 4,
                                    delta = 0.5)
        var = subset["filtered_%s" %orientation].var()
        variances.append(var)
        try:
            if (var) < 1e-2:
                heel_strikes.append(0)
            else:
                heel_strikes.append(len(gp.heel_strikes(subset[orientation])[1]))
        except:
            heel_strikes.append(0)
        jPos = jPos + step_size
        i = i + 1
    
    ## on each time-window chunk collect data into numpy array
    heel_strikes = np.array(heel_strikes)
    variances = np.array(variances)
    ts = pd.DataFrame({"time":time, 
                        "heel_strikes":heel_strikes/(256/100), 
                        "variance": variances})
    
    ## subset data, removing data that has 5 consecutive zero runs ##
    ts = subset_data_non_zero_runs(data = ts, zero_runs_cutoff = 5)
    
    ## store data size after consecutive zero subset
    new_data_size = ts.shape[0]
    if new_data_size == 0:
        return 0
    else:
        mean_heel_strikes_per_chunk = ts["heel_strikes"].mean()
        return mean_heel_strikes_per_chunk
    
    
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
    data[orientation] = butter_lowpass_filter(data = data[orientation], 
                                            sample_rate = 100, 
                                            cutoff=2, 
                                            order=2)
    zcr_list = detect_zero_crossing(data[orientation].values)
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


def get_rotational_features(data, orientation):
    """
    Function to retrieve rotational features
    parameter:
    `data`: pandas dataframe
    return a dataframe containing rotational features
    """
    if isinstance(data, str):
        data = query.open_filepath(data)
        data = query.clean_accelerometer_data(data)
        data = data[data["sensorType"] == "rotationRate"]
    rotation_data = calculate_rotation(data, orientation)
    rotation_data = rotation_data[rotation_data["aucXt"] > 2]
    rotation_dict = {}
    if rotation_data.shape[0] != 0:
        rotation_dict["rotation.no_of_turns"]   = rotation_data.shape[0]
        rotation_dict["rotation.mean_duration"] = rotation_data["turn_duration"].mean()
        rotation_dict["rotation.min_duration"]  = rotation_data["turn_duration"].min()
        rotation_dict["rotation.max_duration"]  = rotation_data["turn_duration"].max()
    else:
        rotation_dict["rotation.no_of_turns"]   = 0
        rotation_dict["rotation.mean_duration"] = 0
        rotation_dict["rotation.min_duration"]  = 0
        rotation_dict["rotation.max_duration"]  = 0
    return rotation_dict



def create_overlay_data(accel_data, rotation_data):
    """
    Function to overlay acceleration data and rotational data
    """
    test = pd.merge(accel_data, rotation_data, on = "td", how = "left")
    test["time"] = test["td"]
    test = test.set_index("time")
    test.index = pd.to_datetime(test.index, unit = "s")
    test["aucXt"] = test["aucXt"].fillna(method = "bfill").fillna(0)
    test["turn_duration"] = test["turn_duration"].fillna(method = "bfill").fillna(0)
    return test


def separate_array_sequence(array):
    """
    function to separate array sequence
    parameter:
    `array`: np.array, or a list
    returns a numpy array groupings of sequences
    """
    seq2 = array
    groups = []
    for _, g in groupby(enumerate(seq2), lambda x: x[0]-x[1]):
        groups.append(list(map(itemgetter(1), g)))
    groups = np.asarray(groups)
    return groups


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
    data = create_overlay_data(accel_ts, rotation_ts)
    data = data.reset_index()
    mean_arr_wchunk = []
    mean_arr_no_wchunk = []
    mean_arr_gait_step_regularity = []
    mean_arr_gait_stride_regularity = []
    mean_arr_gait_symmetry = []
    mean_arr_freq_of_peaks = []
    mean_arr_max_freeze_index = []
    mean_arr_freeze_occ_per_secs = []
    mean_arr_speed_of_gait = []
    walking_seqs = separate_array_sequence(np.where(data["aucXt"]<2)[0])
    for seqs in walking_seqs:
        data_seqs = data.loc[seqs[0]:seqs[-1]].set_index("time")
        no_of_steps_per_secs_wchunk = calculate_number_of_steps_per_window(data = data_seqs, 
                                                                           orientation = orientation)
        mean_arr_wchunk.append(no_of_steps_per_secs_wchunk)
        duration = (data_seqs["td"].iloc[-1] - data_seqs["td"].iloc[0])
        
        ## fixed constants
        gp = pdkit.GaitProcessor(duration = duration,
                                cutoff_frequency = 5,
                                filter_order = 4,
                                delta = 0.5)  
        try:
            no_of_steps_per_secs_no_wchunk = len(gp.heel_strikes(data_seqs[orientation])[0])/duration
        except:
            no_of_steps_per_secs_no_wchunk = 0

        """added all pdkit features"""
        ###### TODO: fix freeze occurences before running the whole data pipeline  #####
        try:
            gait_step_regularity = gp.gait_regularity_symmetry(data_seqs[orientation])[0]
        except:
            gait_step_regularity = 0
        try:
            gait_stride_regularity = gp.gait_regularity_symmetry(data_seqs[orientation])[1]
        except:
            gait_stride_regularity = 0
        try:
            gait_symmetry = gp.gait_regularity_symmetry(data_seqs[orientation])[2]
        except:
            gait_symmetry = 0
        try:
            frequency_of_peaks = gp.frequency_of_peaks(data_seqs[orientation])
        except:
            frequency_of_peaks = 0
        try:
            max_freeze_index = np.max(gp.freeze_of_gait(data_seqs[orientation])[1])
        except:
            max_freeze_index= 0
        try:
            freeze_occurences_per_secs = \
                (sum(i > 2.5 for i in gp.freeze_of_gait(data_seqs[orientation])[1]))/duration
        except:
            freeze_occurences_per_secs = 0
        try:
            speed_of_gait = gp.speed_of_gait(data_seqs[orientation], wavelet_level = 6)
        except:
            speed_of_gait = 0
        #######
        mean_arr_no_wchunk.append(no_of_steps_per_secs_no_wchunk)
        mean_arr_gait_step_regularity.append(gait_step_regularity)
        mean_arr_gait_stride_regularity.append(gait_stride_regularity)
        mean_arr_gait_symmetry.append(gait_symmetry)
        mean_arr_freq_of_peaks.append(frequency_of_peaks)
        mean_arr_max_freeze_index.append(max_freeze_index)
        mean_arr_freeze_occ_per_secs.append(freeze_occurences_per_secs)
        mean_arr_speed_of_gait.append(speed_of_gait)
    # wchunk_mean_no_of_steps_per_secs = np.mean(np.array(mean_arr_wchunk))
    # no_wchunk_mean_no_of_steps_per_secs = np.mean(np.array(mean_arr_no_wchunk))

    feature_dict = {}
    feature_dict["acceleration_fs"] = acceleration_fs
    feature_dict["rotation_fs"] = rotation_fs
    feature_dict["wchunk.mean_no_of_steps_per_secs"] = np.mean(np.array(mean_arr_wchunk))
    feature_dict["no_wchunk.mean_no_of_steps_per_secs"] = np.mean(np.array(mean_arr_no_wchunk))
    feature_dict["mean_gait_step_regularity"] = np.mean(np.array(mean_arr_gait_step_regularity))
    feature_dict["mean_gait_stride_regularity"] = np.mean(np.array(mean_arr_gait_stride_regularity))
    feature_dict["mean_gait_symmetry"] = np.mean(np.array(mean_arr_gait_symmetry))
    feature_dict["mean_freq_of_peaks"] = np.mean(np.array(mean_arr_freq_of_peaks))
    feature_dict["mean_max_freeze_index"] = np.mean(np.array(mean_arr_max_freeze_index))
    feature_dict["mean_freeze_occ_per_secs"] = np.mean(np.array(mean_arr_freeze_occ_per_secs))
    feature_dict["mean_gait_speed"] = np.mean(np.array(mean_arr_speed_of_gait))
    # feature_dict["freeze_indices"] = gp.freeze_of_gait(data_seqs.set_index("time")[orientation])
    if rotation_occurences.shape[0] != 0:
        feature_dict["rotation.no_of_turns"]   = rotation_occurences.shape[0]
        feature_dict["rotation.mean_duration"] = rotation_occurences["turn_duration"].mean()
        feature_dict["rotation.min_duration"]  = rotation_occurences["turn_duration"].min()
        feature_dict["rotation.max_duration"]  = rotation_occurences["turn_duration"].max()
    else:
        feature_dict["rotation.no_of_turns"]   = 0
        feature_dict["rotation.mean_duration"] = 0
        feature_dict["rotation.min_duration"]  = 0
        feature_dict["rotation.max_duration"]  = 0
    return feature_dict

def featurize_wrapper(data):
    """
    wrapper function for multiprocessing jobs
    parameter:
    `data`: takes in pd.DataFrame
    returns a json file featurized data
    """
    data["walk_features"] = data["walk_motion.json_pathfile"].apply(gait_processor_pipeline, orientation = "y")
    return data
