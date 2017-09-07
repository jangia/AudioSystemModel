"""
Created on Fri May 26 17:15:24 2017

@author: jangia
"""
import datetime
import os
from sklearn.externals import joblib
import pandas as pd
import numpy as np
from pymongo import MongoClient
from scipy.fftpack import fft, ifft
from scipy.signal import hann
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

NUM_SAMPLES_IN = 1200
NUM_SAMPLES_OUT = 1200

DATA_RANGE = 962
# create DB connection
client = MongoClient()
db = client.amp

print('Started at: ' + str(datetime.datetime.now()))

# Get all FFTs
fft_ref_all = pd.DataFrame(list(db.fft_ref_amp_phi.find({})))
# fft_all = pd.DataFrame(list(db.fft.find(
#     {
#     'amp': {
#         '$nin': [
#             "1.0",
#             "0.9",
#             "0.81",
#             "0.7290000000000001",
#             "0.09847709021836118",
#             "0.08862938119652507",
#             "0.0717897987691853",
#             "0.07976644307687256"
#         ]
#     }
#     }
# )))
fft_all = pd.DataFrame(list(db.fft_amp_phi.find({'gain': '4', 'volume': '4'})))
print('I have data from database at:' + str(datetime.datetime.now()))

dataset = pd.merge(fft_ref_all, fft_all, how='inner', on=['amp', 'frequency'])

f = dataset.iloc[:, 3].values

# Initialize X
X_amp = np.array([dataset['fft_amp_x'][i][:NUM_SAMPLES_IN] for i in range(0, DATA_RANGE)])

# Initialize real and imag array of output FFT
Y_amp = np.array([dataset['fft_amp_y'][i][:NUM_SAMPLES_OUT] for i in range(0, DATA_RANGE)])

print('X and Y initialized')

regressor_amp = RandomForestRegressor(n_estimators=4, verbose=3, n_jobs=2)
#regressor_phi = RandomForestRegressor(n_estimators=2, verbose=3, n_jobs=2)
print('Fit model at: ' + str(datetime.datetime.now()))
regressor_amp.fit(X_amp, Y_amp)
#regressor_phi.fit(X_phi, Y_phi)
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

fft_data_amp = np.empty([1, NUM_SAMPLES_IN])
fft_data_amp[0][:] = np.abs(fft_data)


# measured out
y_db = dataset[(dataset.frequency == '440.0') & (dataset.volume == '4') & (dataset.gain == '4') & (dataset.amp == str(0.9**6))]
y_meas = np.empty([NUM_SAMPLES_OUT])
y_amp = np.array([y_db['fft_amp_y'].values[0][:NUM_SAMPLES_OUT]])
y_ph = np.array([y_db['fft_ph_y'].values[0][:NUM_SAMPLES_OUT]])

for i in range(0, NUM_SAMPLES_OUT):
    y_meas[i] = y_amp[0][i] * np.cos(y_ph[0][i]) + 1j * y_amp[0][i] * np.sin(y_ph[0][i])

# predicted out
y_pred_phi = [np.unwrap(np.angle(fft_data))]
#y_pred_phi = [[0 for i in range(NUM_SAMPLES_OUT)]]
#y_pred_phi = [[0] for i in range(NUM_SAMPLES_OUT)]
y_pred_amp = regressor_amp.predict(fft_data_amp)


y_pred = np.empty([NUM_SAMPLES_OUT], dtype='complex128')

for i in range(0, NUM_SAMPLES_OUT):
    y_pred[i] = y_pred_amp[0][i] * np.cos(y_pred_phi[0][i]) + 1j * y_pred_amp[0][i] * np.sin(y_pred_phi[0][i])


# draw plots
fig = plt.figure()

plt.subplot(2, 1, 1)
plt.plot(ifft(y_pred), 'b')
plt.title('Modeliran signal na izhodu')
plt.ylabel('Amplituda')
plt.xlabel('Čas')

plt.subplot(2, 1, 2)
plt.title('Originalni signal na izhodu')
plt.plot(ifft(y_meas), 'r')
plt.ylabel('Amplituda')
plt.xlabel('Čas')

filename = 'guitar'
plt.savefig('/home/jangia/Documents/Mag/AudioSystemModel/normal_regression/plots/{0}_{1}.png'.format(filename, datetime.datetime.now()))

plt.close(fig)

# save model
BASE_DIR = os.path.dirname(os.path.abspath('amp_phi_random_forest.py'))
#joblib.dump(regressor_amp, os.path.join(BASE_DIR, 'models', 'test_model_random_forest_with_hann_amp_12000_20trees.pkl'))
#joblib.dump(regressor_phi, os.path.join(BASE_DIR, 'models', 'test_model_random_forest_with_hann_phi.pkl'))

print('Finished at: ' + str(datetime.datetime.now()))
