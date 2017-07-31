"""
Created on Fri May 26 17:15:24 2017

@author: jangia
"""
import datetime
import json
import os
import random
from sklearn.externals import joblib
import pandas as pd
import numpy as np
from pymongo import MongoClient
from scipy.fftpack import fft
from scipy.signal import hann
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

NUM_SAMPLES_IN = 12000
NUM_SAMPLES_OUT = 12000

DATA_RANGE = 5324
# create DB connection
client = MongoClient()
db = client.amp

print('Started at: ' + str(datetime.datetime.now()))

# Get all FFTs
fft_ref_all = pd.DataFrame(list(db.fft_ref.find({})))
fft_all = pd.DataFrame(list(db.fft.find(
    {
    'amp': {
        '$nin': [
            "1.0",
            "0.9",
            "0.81",
            "0.7290000000000001",
            "0.09847709021836118",
            "0.08862938119652507",
            "0.0717897987691853",
            "0.07976644307687256"
        ]
    }
    }
)))

print('I have data from database at:' + str(datetime.datetime.now()))

dataset = pd.merge(fft_all, fft_ref_all, how='inner', on=['amp', 'frequency'])

f = dataset.iloc[:, 3].values

# Set amplitude and input FFT as input data
gain = dataset.iloc[:, -6].values.astype('float64')
volume = dataset.iloc[:, -4].values.astype('float64')

# Initialize X
X_raw_re = dataset.iloc[:, -1].values
X_raw_im = dataset.iloc[:, -2].values

# Initialize X
X_amp = np.empty([DATA_RANGE, NUM_SAMPLES_IN+2])
X_phi = np.empty([DATA_RANGE, NUM_SAMPLES_IN+2])

# Set output FFT
Y_raw_re = dataset.iloc[:, 3].values
Y_raw_im = dataset.iloc[:, 2].values

# Initialize real and imag array of output FFT
Y_amp = np.empty([DATA_RANGE, NUM_SAMPLES_OUT])
Y_phi = np.empty([DATA_RANGE, NUM_SAMPLES_OUT])

print('X and Y initialized')
print('Filling X and Y with values from database' + str(datetime.datetime.now()))
# Convert from real and imag to phase and amplitude
for i in range(0, DATA_RANGE):

    X_amp[i][0] = gain[i]
    X_amp[i][1] = volume[i]
    X_phi[i][0] = gain[i]
    X_phi[i][1] = volume[i]

    for j in range(0, max(NUM_SAMPLES_OUT, NUM_SAMPLES_IN)):

        if j < NUM_SAMPLES_OUT:
            y_i = Y_raw_re[i][j] + 1j * Y_raw_im[i][j]
            Y_amp[i][j] = np.abs(y_i)
            Y_phi[i][j] = np.unwrap(np.angle(np.array([y_i])))

        if j < NUM_SAMPLES_IN:
            x_i = X_raw_re[i][j] + 1j * X_raw_im[i][j]
            X_amp[i][j + 2] = np.abs(x_i)
            X_phi[i][j + 2] = np.unwrap(np.angle(np.array([x_i])))

print('X and Y filled: ' + str(datetime.datetime.now()))

regressor_amp = RandomForestRegressor(n_estimators=4, verbose=3, n_jobs=4)
regressor_phi = RandomForestRegressor(n_estimators=4, verbose=3, n_jobs=4)
print('Fit model at: ' + str(datetime.datetime.now()))
regressor_amp.fit(X_amp, Y_amp)
regressor_phi.fit(X_phi, Y_phi)
print('Model fitted at: ' + str(datetime.datetime.now()))

# Predicting the Test set results
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath('audio_processing.py')))
# REC_PATH = os.path.join(BASE_DIR, 'guitar', 'sine.wav')
# rate, audio_data = wav.read(os.path.join(BASE_DIR, REC_PATH))

FS = 96000
T_END = 1

t = np.arange(FS * T_END)
chunk = 0.5 * np.sin(2 * np.pi * 440 * t / FS)
fft_data = fft(chunk[:24000] * hann(24000))[:NUM_SAMPLES_IN]

# fft_data = fft(audio_data[40000:60000])[:10000]

fft_data_amp = np.empty([1, NUM_SAMPLES_IN + 2])
fft_data_amp[0][0] = np.float64(4)
fft_data_amp[0][1] = np.float64(4)
fft_data_amp[0][2:] = np.abs(fft_data)

fft_data_phi = np.empty([1, NUM_SAMPLES_IN + 2])
fft_data_phi[0][0] = np.float64(4)
fft_data_phi[0][1] = np.float64(4)
fft_data_phi[0][2:] = np.unwrap(np.angle(fft_data))

# predicted out
y_pred_phi = regressor_phi.predict(fft_data_phi)
y_pred_amp = regressor_amp.predict(fft_data_amp)

y_pred = np.empty([NUM_SAMPLES_OUT], dtype='complex128')

for i in range(0, NUM_SAMPLES_OUT):
    y_pred[i] = y_pred_amp[0][i] * np.cos(y_pred_phi[0][i]) + 1j * y_pred_amp[0][i] * np.sin(y_pred_phi[0][i])

# measured out
y_db = dataset[(dataset.frequency == '440.0') & (dataset.volume == '4') & (dataset.gain == '4') & (dataset.amp == str(0.9**6))]
y_meas = np.empty([NUM_SAMPLES_OUT])
y_raw_re = y_db.iloc[:, 3].values
y_raw_im = y_db.iloc[:, 2].values

for j in range(0, NUM_SAMPLES_OUT):
    y_meas[j] = np.abs(y_raw_re[0][j] + 1j * y_raw_im[0][j])

# draw plots
fig = plt.figure()

plt.subplot(2, 1, 1)
plt.semilogy(abs(y_meas), 'r')
plt.title('Original vs Modeled')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.semilogy(abs(y_pred), 'b')
plt.title('Original vs Modeled')
plt.ylabel('Amplitude')

filename = 'guitar'
plt.savefig('/home/jangia/Documents/Mag/AudioSystemModel/normal_regression/plots/{0}_{1}.png'.format(filename, datetime.datetime.now()))

plt.close(fig)

# save model
BASE_DIR = os.path.dirname(os.path.abspath('amp_phi_random_forest.py'))
joblib.dump(regressor_amp, os.path.join(BASE_DIR, 'models', 'test_model_random_forest_with_hann_amp.pkl'))
joblib.dump(regressor_phi, os.path.join(BASE_DIR, 'models', 'test_model_random_forest_with_hann_phi.pkl'))

print('Finished at: ' + str(datetime.datetime.now()))
