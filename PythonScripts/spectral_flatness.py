import pandas as pd
import numpy as np 
import scipy as sp
import scipy.fftpack
import json
import statsmodels as stats
from sklearn.metrics import auc
from myutils import get_acceleration_ts

def get_spectrum(signal):
    ## Number of samplepoints
    N = signal.shape[0]
    ## sample spacing 1/sampling rate delta t
    T = 1/100
    ## linspace of 0 to 20 seconds given N samples
    x = np.linspace(0.0, N*T, N)
    y = signal 
    ## calculate fft
    yf = scipy.fftpack.fft(y)
    xf = np.linspace(0.0, 1.0/(2.0 * T), N/2)
    magnitude = 2.0/N * np.abs(yf[:N//2])
    ## return spectrum
    return pd.DataFrame({"Frequency": xf, "Magnitude": magnitude})

def sfm(data, start, end, gm_range):
    arr = []
    for gamma in gm_range:
        data = data[(data["Frequency"] >= start) & (data["Frequency"] <= end)]
        data["Magnitude"] =  data["Magnitude"]/ data["Magnitude"].max()
        pdf =  (np.abs(data["Magnitude"] ** (1/gamma)))
        ### geometric mean with gamma correction
        geo_mean = sp.stats.gmean(pdf)
        ### arithmetic mean with gamma correction
        ar_mean = pdf.mean()
        ### calculation of spectral flatness
        spectral_flatness = geo_mean/ar_mean
        ### append spectral flatness result
        arr.append(spectral_flatness)
    return pd.DataFrame({"gamma": gm_range, "sfm":arr})

def sfm_preprocessing(data, hz_start = 0, hz_end = 10, gamma_range = np.arange(0.2, 2, 0.01)):
    spec_data = get_spectrum(data)
    sfm_data = sfm(spec_data, hz_start, hz_end, gamma_range)
    return sfm_data

def sfm_auc_pipeline(params, var):
    ## process acceleration
    try:
        data = get_acceleration_ts(params)
        # print(data)
    except:
        return "#ERROR"
    ### if filepath is empty or have no accelerometer data ###
    if isinstance(data, str):
        return data
    ## run spectral flatness pipeline
    data = sfm_preprocessing(data[var])
    # print(data)
    ## retrieve AUC
    area = auc(data.gamma, data.sfm)
    return area

def sfm_featurize(data):
    for coord in ["x", "y", "z", "AA"]:
        for pathfile in [_ for _ in data.columns if ("pathfile" in _) 
                                                and ("balance" in _ or "rest" in _ )]:
            print(pathfile)
            data["sfm_auc_{}".format(coord)] = data[pathfile].apply(sfm_auc_pipeline, var = coord)
            # print(data)
    return data