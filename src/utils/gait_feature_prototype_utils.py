import sys
import pdkit
import synapseclient as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import utils.query_utils as query
import utils.new_gait_feature_utils as gproc
from scipy import signal
import warnings
from scipy.fftpack import (rfft, fftfreq)
from scipy.signal import (butter, lfilter, correlate, freqz)
from sklearn import metrics
from operator import itemgetter
import time
from itertools import *
from sklearn import metrics
import time
warnings.simplefilter("ignore")


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
        ts_arr.append({"axis": orientation, 
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

    rotation_ts = calculate_rotation(rotation_ts, "y")
    data = create_overlay_data(accel_ts, rotation_ts)
    data = data.reset_index()
    walking_seqs = separate_array_sequence(np.where(data["aucXt"]<2)[0])
    gait_feature_arr = []
    for seqs in walking_seqs:
        data_seqs = data.loc[seqs[0]:seqs[-1]].set_index("time")
        gait_feature_arr.append(compute_gait_feature_per_window(data = data_seqs, 
                                                                orientation = orientation))
    return [j for i in gait_feature_arr for j in i]


def detect_zero_crossing(array):
    """
    Function to detect zero crossings in a time series signal data
    parameter:
        `array`: numpy array
    returns index location before sign change 
    """
    zero_crossings = np.where(np.diff(np.sign(array)))[0]
    return zero_crossings

def calculate_rotation(data, orientation):
    """
    function to calculate rotational movement gyroscope AUC * period of zero crossing
    note: filter order of 2 is used based on research paper (common butterworth filter), 
            cutoff frequency of 2 is used to smoothen the signal for recognizing 
            area under the curve more
    parameter:
        `data`       : pandas dataframe
        `orientation`: orientation (string)
    
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
    list_rotation = []
    turn_window = 0
    for i in zcr_list: 
        x = data["td"].iloc[start:i+1].values
        y = data[orientation].iloc[start:i+1].values
        turn_duration = data["td"].iloc[i+1] - data["td"].iloc[start]
        start  = i + 1
        if (len(y) >= 2):
            auc   = np.abs(metrics.auc(x,y)) 
            aucXt = auc * turn_duration
            omega = auc / turn_duration
            if aucXt > 2:
                turn_window += 1
                list_rotation.append({
                    "turn_start": x[0],
                    "turn_end":  x[-1],
                    "axis": orientation,
                    "turn_duration": turn_duration,
                    "auc": auc, ## radian
                    "omega": omega, ## radian/secs 
                    "aucXt":aucXt, ## radian . secs (based on research paper)
                    "turn_window": turn_window
                })
    if len(list_rotation) == 0:
        return "#ERROR"
    return list_rotation

def compute_rotational_features(filepath, orientation):
    rotation_ts = query.get_sensor_ts_from_filepath(filepath = filepath, 
                                                    sensor = "rotationRate")
    if not isinstance(rotation_ts, pd.DataFrame):
        return "#ERROR"
    rotation_ts = calculate_rotation(rotation_ts, "y")
    return rotation_ts

def pdkit_gait_featurize_wrapper(data):
    """
    wrapper function for multiprocessing jobs
    parameter:
    `data`: takes in pd.DataFrame
    returns a json file featurized data
    """
    data["gait.pdkit_features"] = data["walk_motion.json_pathfile"].apply(gait_processor_pipeline, orientation = "y")
    return data

def rotation_gait_featurize_wrapper(data):
    data["gait.rotational_features"] = data["walk_motion.json_pathfile"].apply(compute_rotational_features, orientation = "y")
    return data


