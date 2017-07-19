import datetime
import json
from collections import OrderedDict

import pandas as pd
import numpy as np
from numpy.fft import ifft
from pymongo import MongoClient
from keras.models import load_model
import matplotlib.pyplot as plt


model_real = load_model('/home/jangia/Documents/Mag/AudioSystemModel/neural_network/models/test_model_real.h5')
model_imag = load_model('/home/jangia/Documents/Mag/AudioSystemModel/neural_network/models/test_model_imag.h5')

cnt = 0
NUM_SAMPLES_IN = 1200
NUM_SAMPLES_OUT = 1200

DATA_RANGE = 56
# create DB connection
client = MongoClient()
db = client.amp


cnt = 0


print('Started at: ' + str(datetime.datetime.now()))

# Get all FFTs
fft_ref_all = pd.DataFrame(list(db.fft_ref.find({})))
fft_all = pd.DataFrame(list(db.fft.find({})))

print('I have data from database at:' + str(datetime.datetime.now()))

dataset = pd.merge(fft_all, fft_ref_all, how='inner', on=['amp', 'frequency'])
dataset = dataset.sample(frac=1).reset_index(drop=True)

f = dataset.iloc[:, 4].values

# Set amplitude and input FFT as input data
gain = np.float64(dataset.iloc[:, -6].values)[0]
volume = np.float64(dataset.iloc[:, -4].values)[0]

X3 = dataset.iloc[:, -1].values

# Initialize X
X_re = dataset.iloc[:, -1].values
X_im = dataset.iloc[:, -2].values

fig = plt.figure()
# Set output FFT
Y_re = dataset.iloc[:, 3].values
Y_im = dataset.iloc[:, 2].values

Y = np.empty([NUM_SAMPLES_OUT], dtype='complex128')
X = np.empty([NUM_SAMPLES_IN], dtype='complex128')

Y_no_phi = np.empty([len(Y_im[0])], dtype='complex128')

for i in range(0, NUM_SAMPLES_OUT):
    Y[i] = Y_re[0][i] + 1j * Y_im[0][i]
    X[i] = X_re[0][i] + 1j * X_im[0][i]

print(max(Y))

for i in range(0, NUM_SAMPLES_IN):
    Y_no_phi[i] = abs(X[i]) * np.cos(0) + 1j * abs(X[i]) * np.sin(0)

print(len(X))
y = ifft(Y_no_phi)
y = np.concatenate([y[600:1200], y[0:600]])
x = ifft(X)

plt.subplot(2, 1, 1)
plt.plot(np.unwrap(np.angle(X)), 'r')
plt.title('Measured Output VS Predicted Output')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.plot(x, 'b')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Amplitude')

# plt.subplot(2, 1, 1)
# plt.semilogy(abs(Y), 'r')
# plt.title('Measured Output VS Predicted Output')
# plt.ylabel('Amplitude')
#
# plt.subplot(2, 1, 2)
# plt.semilogy(abs(X))
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Amplitude')


plt.show()

print('End')
