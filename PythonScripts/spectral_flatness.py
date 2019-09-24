import pandas as pd
import numpy as np 
import scipy as sp
import scipy.fftpack
import json
import statsmodels as stats
from sklearn.metrics import auc
from utils import processAcceleration

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

def run_sfm_pipeline(data, hz_start = 1, hz_end = 4, gamma_range = np.arange(0.2, 5, 0.01)):
    spec_data = get_spectrum(data)
    sfm_data = sfm(spec_data, hz_start, hz_end, gamma_range)
    return sfm_data

def sfm_auc(params, var):
    
    
    ## process acceleration
    data = processAcceleration(params)
    
    ### if filepath is empty ###
    if data == "EMPTY FILEPATHS":
        return data
    
    ### if filepaths are not empty but accelerometer data is empty ###
    elif data == "NO ACCELEROMETER DATA":
        return data
    
    ## run spectral flatness pipeline
    data = run_sfm_pipeline(data[var])
    
    ## retrieve AUC
    area = auc(data.gamma, data.sfm)
    
    return area