import datetime
import os

import numpy as np
# Predicting the Test set results
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath('audio_processing.py')))
# REC_PATH = os.path.join(BASE_DIR, 'guitar', 'sine.wav')
# rate, audio_data = wav.read(os.path.join(BASE_DIR, REC_PATH))
from pymongo import MongoClient
from scipy.fftpack import fft, ifft
from scipy.signal import hann
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.externals import joblib

from config import load_config

config = load_config()
NUM_SAMPLES_IN = config['small_models']['num_samples_in']
NUM_SAMPLES_OUT = config['small_models']['num_samples_out']
NUM_ESTIMATORS = config['random_forest']['small_model']['num_of_estimators']
NUM_JOBS = config['random_forest']['small_model']['num_of_jobs']
FS = 96000
T_END = 1

# create DB connection
client = MongoClient()
collection_ref = client.amp.fft_ref_amp_phi
collection = client.amp.fft_amp_phi

print('Started at: ' + str(datetime.datetime.now()))

# Get all FFTs
fft_ref_all = pd.DataFrame(list(collection_ref.find({})))
fft_all = pd.DataFrame(list(collection.find({'gain': '4', 'volume': '4'})))
print('I have data from database at:' + str(datetime.datetime.now()))

dataset = pd.merge(fft_ref_all, fft_all, how='inner', on=['amp', 'frequency'])

# create signal
t = np.arange(FS * T_END)
chunk = 0.5 * np.sin(2 * np.pi * 440 * t / FS)
fft_data = fft(chunk[:24000] * hann(24000))[:NUM_SAMPLES_IN]

fft_data_amp = np.empty([1, NUM_SAMPLES_IN])
fft_data_amp[0][:] = np.abs(fft_data)

# measured out
y_db = dataset[(dataset.frequency == '440.0') & (dataset.volume == '4') & (dataset.gain == '4') & (dataset.amp == str(0.9**6))]
y_meas = np.empty([NUM_SAMPLES_OUT])
y_amp = np.array([y_db['fft_amp_y'].values[0][:NUM_SAMPLES_OUT]])
y_ph = np.array([y_db['fft_ph_y'].values[0][:NUM_SAMPLES_OUT]])

for i in range(0, NUM_SAMPLES_OUT):
    y_meas[i] = y_amp[0][i] * np.cos(y_ph[0][i]) + 1j * y_amp[0][i] * np.sin(y_ph[0][i])

# load model
MODELS_PATH = os.path.join(os.path.dirname(os.path.abspath('config.py')), config['models_location'])
regressor_amp = joblib.load(
    os.path.join(MODELS_PATH, config['random_forest']['models_location'], 'random_forest_small.pkl')
)
# predicted out
y_pred_phi = [np.unwrap(np.angle(fft_data))]
y_pred_amp = regressor_amp.predict(fft_data_amp)

y_pred = np.empty([NUM_SAMPLES_OUT], dtype='complex128')

for i in range(0, NUM_SAMPLES_OUT):
    y_pred[i] = y_pred_amp[0][i] * np.cos(y_pred_phi[0][i]) + 1j * y_pred_amp[0][i] * np.sin(y_pred_phi[0][i])

PLOT_PATH = os.path.join(os.path.dirname(os.path.abspath('test_polynomial_small.py')), 'plots')

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
plt.savefig(os.path.join(PLOT_PATH, '{0}_{1}.png'.format(filename, datetime.datetime.now())))

plt.close(fig)
