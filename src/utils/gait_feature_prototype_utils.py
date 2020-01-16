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

from scipy import interpolate, signal, fft
from scipy.fftpack import rfft
from pywt import wavedec

from pdkit.utils import (load_data,
                        numerical_integration, 
                        autocorrelation,
                        peakdet,
                        compute_interpeak,
                        crossings_nonzero_pos2neg,
                        autocorrelate,
                        get_signal_peaks_and_prominences,
                        BellmanKSegment)


warnings.simplefilter("ignore")


### referenced from pdkit butter low pass filters ##
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
        print(i)
        data = data.loc[~data.index.isin(range(i[0],i[1]))]
    return data.reset_index(drop = True)

def zero_runs(array):
    """
    Function to search zero runs in an np.array
    
    parameter:
        `array`  : array of sequence (type: np.array or list)
    
    returns:
         N x 2 np.array matrix containing zero runs
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
        `array`: array of number sequence (type = np.array or list)
    
    returns: 
        index location before sign change in sequence (type = N x 2 np.array)
        format: [[index start, index_end], 
                [[index start, index_end], .....]
    """
    zero_crossings = np.where(np.diff(np.sign(array)))[0]
    return zero_crossings

def compute_rotational_features(accel_data, rotation_data):
    """
    function to calculate rotational movement gyroscope AUC * period of zero crossing
    note: filter order of 2 is used based on research paper (common butterworth filter), 
            cutoff frequency of 2 is used to smoothen the signal for recognizing 
            area under the curve more
    parameter:
        `accel_data`   : (timeIndex(index), td (timeDifference), y, z, AA) accelerometer data (type = pd.DataFrame)
        `rotation_data : (timeIndex(index), td (timeDifference), y, z, AA) gyroscope data (type = pd.DataFrame)
    
    returns: 
        list of dictionary of gait features during rotational motions
        format: [{omega: some_value, ....}, 
                {omega: some_value, ....},....]
        
    """
    list_rotation = []
    for orientation in ["x", "y", "z", "AA"]:
        start = 0
        dict_list = {}
        dict_list["td"] = []
        dict_list["auc"] = []
        dict_list["turn_duration"] = []
        dict_list["aucXt"] = []
        rotation_data[orientation] = butter_lowpass_filter(data = rotation_data[orientation], 
                                                            sample_rate = 100, 
                                                            cutoff = 2, 
                                                            order = 2) 
        zcr_list = detect_zero_crossing(rotation_data[orientation].values)
        turn_window = 0
        for i in zcr_list: 
            x_rot = rotation_data["td"].iloc[start:i+1]
            y_rot = rotation_data[orientation].iloc[start:i+1]
            y_accel = accel_data[orientation].iloc[start:i+1]
            turn_duration = rotation_data["td"].iloc[i+1] - rotation_data["td"].iloc[start]
            start  = i + 1
            if (len(y_rot) >= 2):
                auc   = np.abs(metrics.auc(x_rot,y_rot)) 
                aucXt = auc * turn_duration
                omega = auc / turn_duration
                if aucXt > 2:
                    turn_window += 1
                    gp = pdkit.GaitProcessor(duration = turn_duration,
                                            cutoff_frequency = 5,
                                            filter_order = 4,
                                            delta = 0.5)
                    try:
                        strikes, _ = gp.heel_strikes(y_accel)
                        steps      = np.size(strikes) 
                        cadence    = steps/turn_duration
                    except:
                        steps   = 0
                        cadence = 0
                    try:
                        peaks_data = y_accel.values
                        maxtab, _ = peakdet(peaks_data, gp.delta)
                        x = np.mean(peaks_data[maxtab[1:,0].astype(int)] - peaks_data[maxtab[:-1,0].astype(int)])
                        frequency_of_peaks = abs(1/x)
                    except:
                        frequency_of_peaks = 0
                    if steps >= 2:   # condition if steps are more than 2, during 2.5 seconds window 
                        step_durations = []
                        for i in range(1, np.size(strikes)):
                            step_durations.append(strikes[i] - strikes[i-1])
                        avg_step_duration = np.mean(step_durations)
                        sd_step_duration = np.std(step_durations)
                    else:
                        avg_step_duration = 0
                        sd_step_duration = 0

                    if steps >= 4:
                        strides1 = strikes[0::2]
                        strides2 = strikes[1::2]
                        stride_durations1 = []
                        for i in range(1, np.size(strides1)):
                            stride_durations1.append(strides1[i] - strides1[i-1])
                        stride_durations2 = []
                        for i in range(1, np.size(strides2)):
                            stride_durations2.append(strides2[i] - strides2[i-1])
                        strides = [strides1, strides2]
                        stride_durations = [stride_durations1, stride_durations2]
                        avg_number_of_strides = np.mean([np.size(strides1), np.size(strides2)])
                        avg_stride_duration = np.mean((np.mean(stride_durations1),
                                    np.mean(stride_durations2)))
                        sd_stride_duration = np.mean((np.std(stride_durations1),
                                    np.std(stride_durations2)))
                    else:
                        avg_number_of_strides = 0
                        avg_stride_duration = 0
                        sd_stride_duration = 0
                    
                    list_rotation.append({
                            "rotation.axis"                 : orientation,
                            "rotation.energy_freeze_index"  : calculate_freeze_index(y_accel)[0],
                            "rotation.turn_duration"        : turn_duration,
                            "rotation.auc"                  : auc,      ## radian
                            "rotation.omega"                : omega,    ## radian/secs 
                            "rotation.aucXt"                : aucXt,    ## radian . secs (based on research paper)
                            "rotation.window_start"         : x_rot[0],
                            "rotation.window_end"           : x_rot[-1],
                            "rotation.num_window"           : turn_window,
                            "rotation.avg_step_duration"    : avg_step_duration,
                            "rotation.sd_step_duration"     : sd_step_duration,
                            "rotation.steps"                : steps,
                            "rotation.cadence"              : cadence,
                            "rotation.frequency_of_peaks"   : frequency_of_peaks,
                            "rotation.avg_number_of_strides": avg_number_of_strides,
                            "rotation.avg_stride_duration"  : avg_stride_duration,
                            "rotation.sd_stride_duration"   : sd_stride_duration
                    })
    return list_rotation


def split_dataframe_to_dict_chunk_by_interval(accel_data, rotation_data):
    """
    A function to separate dataframe to several chunks separated by rotational motion
    done by a subject. 
    parameter:
        `accel_data`   : accelerometer data (pd.DataFrame)
        `rotation_data`: rotation data (pd.DataFrame)
    
    returns: 
        A dictionary mapping of data chunks of non-rotational motion
    
    format: {"chunk1": pd.DataFrame, 
            "chunk2": pd.DataFrame, etc ......}
    """
    if (not isinstance(accel_data, pd.DataFrame)):
        raise Exception("please use dataframe for acceleration")
    data_chunk = {}
    window = 1 
    last_stop = 0
    #if no rotation#
    if len(rotation_data) == 0 :
        data_chunk["chunk1"] = accel_data
        return data_chunk
    rotation_data = pd.DataFrame(rotation_data)
    for start, end in rotation_data[["rotation.window_start", "rotation.window_end"]].values:
        if last_stop > start:
            raise Exception("Rotational sequence is overlapping or distorted")  
        ## edge case -> rotation starts at zero ##
        if start <= 0:
            last_stop = end
            continue
        ## edge case -> if rotation is overlapping ##
        if last_stop == start:
            last_stop = end
            continue
        ## ideal case ## 
        data_chunk["chunk%s"%window] = accel_data[(accel_data["td"]<=start) & (accel_data["td"]>=last_stop)]
        last_stop = end
        window += 1
    ## edge case -> after the last rotation ## 
    if last_stop < accel_data["td"][-1]:
        data_chunk["chunk%s"%str(window)] = accel_data[(accel_data["td"]>=end)]
    return data_chunk

def compute_pdkit_feature_per_window(data):
    """
    A modified function to calculate feature per smaller time window chunks
    parameter: 
        `data`        : data (type = pd.DataFrame)
        `orientation` : coordinate orientation of the time series (type = string)
    
    returns:
         returns list of dict of walking features 
    
    format: [{steps: some_value, ...}, 
            {steps: some_value, ...}, ...]
    
    """
    ts = data.copy()
    window_size = 256
    step_size   = 50
    jPos        = window_size + 1
    ts_arr      = []
    i           = 0
    if len(ts) < jPos:
        print(ts.shape[0]/ts["td"][-1])
        ts_arr.append(generate_pdkit_features_in_dict(ts))
        return ts_arr
    while jPos < len(ts):
        jStart = jPos - window_size
        subset = ts.iloc[jStart:jPos]
        ts_arr.append(generate_pdkit_features_in_dict(subset)) 
        jPos += step_size
        i = i + 1
    return [j for i in ts_arr for j in i]

def generate_pdkit_features_in_dict(data):
    """
    Function to generate pdkit features given orientation and time-series dataframe
    
    -- Featurization side-note --
    >> Low-variance data will be removed
    >> TODO: sampling frequency will be capped at 100 as sometimes having >100Hz causes some issue in the package
    >> It will try and catch every of the pdkit features, an error in the process will be catch as zero value

    parameter:
        `data`       : dataframe of time series (timeIndex, x, y, z, AA, td)
        `orientation : axis oriention of walking (type: str)

    returns:
         a dictionary mapping of pdkit features 

    """

    feature_list = []
    for orientation in ["x", "y", "z", "AA"]:
        window_duration = data.td[-1] - data.td[0]
        sample_rate = data.shape[0]/window_duration
        y_accel = data[orientation]
        var = y_accel.var()
        gp = pdkit.GaitProcessor(duration = window_duration,
                            cutoff_frequency = 5,
                            filter_order = 4,
                            delta = 0.5, 
                            sampling_frequency = 100)
        try:
            if (var) < 1e-3:
                steps = 0
                cadence = 0
            else:
                strikes, _ = gp.heel_strikes(y_accel)
                steps      = np.size(strikes) 
                cadence    = steps/window_duration
        except:
            steps = 0  
            cadence = 0
        try:
            peaks_data = y_accel.values
            maxtab, _ = peakdet(peaks_data, gp.delta)
            x = np.mean(peaks_data[maxtab[1:,0].astype(int)] - peaks_data[maxtab[:-1,0].astype(int)])
            frequency_of_peaks = abs(1/x)
        except:
            frequency_of_peaks = 0

        try:
            speed_of_gait = gp.speed_of_gait(y_accel, wavelet_type='db3', wavelet_level=6)   
        except:
            speed_of_gait = 0
        if steps >= 2:   # condition if steps are more than 2, during 2.5 seconds window 
            step_durations = []
            for i in range(1, np.size(strikes)):
                step_durations.append(strikes[i] - strikes[i-1])
            avg_step_duration = np.mean(step_durations)
            sd_step_duration = np.std(step_durations)
        else:
            avg_step_duration = 0
            sd_step_duration = 0

        if steps >= 4:
            strides1 = strikes[0::2]
            strides2 = strikes[1::2]
            stride_durations1 = []
            for i in range(1, np.size(strides1)):
                stride_durations1.append(strides1[i] - strides1[i-1])
            stride_durations2 = []
            for i in range(1, np.size(strides2)):
                stride_durations2.append(strides2[i] - strides2[i-1])
            strides = [strides1, strides2]
            stride_durations = [stride_durations1, stride_durations2]
            avg_number_of_strides = np.mean([np.size(strides1), np.size(strides2)])
            avg_stride_duration = np.mean((np.mean(stride_durations1),
                        np.mean(stride_durations2)))
            sd_stride_duration = np.mean((np.std(stride_durations1),
                        np.std(stride_durations2)))
        else:
            avg_number_of_strides = 0
            avg_stride_duration = 0
            sd_stride_duration = 0

        try:
            step_regularity   = gp.gait_regularity_symmetry(y_accel,average_step_duration=avg_step_duration, 
                                                        average_stride_duration=avg_stride_duration)[0]
        except:
            step_regularity   = 0
        
        try:
            stride_regularity = gp.gait_regularity_symmetry(y_accel,average_step_duration=avg_step_duration, 
                                                        average_stride_duration=avg_stride_duration)[1]
        except:
            stride_regularity = 0     

        try:
            symmetry          =  gp.gait_regularity_symmetry(y_accel,average_step_duration=avg_step_duration, 
                                                        average_stride_duration=avg_stride_duration)[2]       
        except:
            symmetry          = 0                                                                                                             
        
        feature_list.append({
                "walking.window_duration"      : window_duration,
                "walking.axis"                 : orientation,
                "walking.energy_freeze_index"  : calculate_freeze_index(y_accel)[0],
                "walking.avg_step_duration"    : avg_step_duration,
                "walking.sd_step_duration"     : sd_step_duration,
                "walking.steps"                : steps,
                "walking.cadence"              : cadence,
                "walking.frequency_of_peaks"   : frequency_of_peaks,
                "walking.avg_number_of_strides": avg_number_of_strides,
                "walking.avg_stride_duration"  : avg_stride_duration,
                "walking.sd_stride_duration"   : sd_stride_duration,
                "walking.speed_of_gait"        : speed_of_gait,
                "walking.step_regularity"      : step_regularity,
                "walking.stride_regularity"    : stride_regularity,
                "walking.symmetry"             : symmetry
        })
    return feature_list

def calculate_freeze_index(data):
    """
    modified pdkit FoG function, removed resampling the signal
    
    parameters: 
        `data`: pd.Series of signal in one orientation
    
    returns:
        array of [freeze index , sumLocoFreeze]
    """
    loco_band=[0.5, 3]
    freeze_band=[3, 8]
    window_size = 256
    sampling_frequency = 100
    f_res = sampling_frequency / window_size
    f_nr_LBs = int(loco_band[0] / f_res)
    f_nr_LBe = int(loco_band[1] / f_res)
    f_nr_FBs = int(freeze_band[0] / f_res)
    f_nr_FBe = int(freeze_band[1] / f_res)
    data = data.values - np.mean(data.values)
    
    Y = np.fft.fft(data, int(window_size))
    Pyy = abs(Y*Y) / window_size
    areaLocoBand = numerical_integration( Pyy[f_nr_LBs-1 : f_nr_LBe], sampling_frequency)
    areaFreezeBand = numerical_integration( Pyy[f_nr_FBs-1 : f_nr_FBe], sampling_frequency)
    
    sumLocoFreeze = areaFreezeBand + areaLocoBand
    freezeIndex = areaFreezeBand / areaLocoBand
    
    return freezeIndex, sumLocoFreeze
    



def walk_feature_pipeline(filepath):
    """
    Function of data pipeline for subsetting data from rotational movements, retrieving rotational features, 
    removing low-variance longitudinal data and PDKIT estimation of heel strikes based on 2.5 secs window chunks
    parameters:
    
        `filepath`    : string of filepath to /.synapseCache (type = str)
        `orientation` : orientation of featurized data (type = str)
    
    returns: 
        gait feature as a series in the dataframe
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
    rotation_occurence = compute_rotational_features(accel_ts, rotation_ts)
    gait_dict = split_dataframe_to_dict_chunk_by_interval(accel_ts, rotation_occurence)
    gait_feature_arr = []
    for chunks in gait_dict.keys():
        gait_feature_arr.append(compute_pdkit_feature_per_window(data = gait_dict[chunks]))
    return [j for i in gait_feature_arr for j in i]


def rotation_feature_pipeline(filepath):
    rotation_ts = query.get_sensor_ts_from_filepath(filepath, "rotationRate")
    accel_ts = query.get_sensor_ts_from_filepath(filepath, "userAcceleration")                                  
    if not isinstance(rotation_ts, pd.DataFrame):
        return "#ERROR"
    rotation_ts = compute_rotational_features(accel_ts, rotation_ts)
    if len(rotation_ts) == 0:
        return "#ERROR"
    return rotation_ts

def annotate_consecutive_zeros(data, feature):
    """
    Function to annotate consecutive zeros in a dataframe

    parameter:
        `data`    : dataframe
        `feature` : feature to assess on counting consecutive zeros
    
    returns:
        A new column-series of data with counted consecutive zeros (if available)
    """
    step_shift_measure = data[feature].ne(data[feature].shift()).cumsum()
    counts = data.groupby(['recordId', step_shift_measure])[feature].transform('size')
    data['consec_zero_steps_count'] = np.where(data[feature].eq(0), counts, 0)
    return data

def walk_featurize_wrapper(data):
    """
    wrapper function for walking multiprocessing jobs
    parameter:
        `data`: takes in pd.DataFrame
    returns a json file featurized walking data
    """
    data["gait.walk_features"] = data["gait.json_pathfile"].apply(walk_feature_pipeline)
    return data

def rotation_featurize_wrapper(data):
    """
    wrapper function for rotation multiprocessing jobs
    parameter:
        `data`: takes in pd.DataFrame
    returns a json file featurized rotation data
    """
    data["gait.rotation_features"] = data["gait.json_pathfile"].apply(rotation_feature_pipeline)
    return data


