"""
Created on Fri May 26 17:15:24 2017

@author: jangia
"""
import datetime
import json
import os
import random

import pandas as pd
import numpy as np
from pymongo import MongoClient
from keras.models import Sequential
from keras.layers import Dense, Activation, GaussianNoise
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft
from scipy.io import wavfile as wav
from keras.layers.advanced_activations import LeakyReLU, ELU, ThresholdedReLU
from scipy.signal import hann

from config import load_config

config = load_config()
NUM_SAMPLES_IN = config['small_models']['num_samples_in']
NUM_SAMPLES_OUT = config['small_models']['num_samples_out']

# create DB connection
client = MongoClient()
collection = client.amp.fft_amp_phi_guitar_solo

print('Started at: ' + str(datetime.datetime.now()))

# Get all FFTs
dataset = pd.DataFrame(list(collection.find({})))
print('I have data from database at:' + str(datetime.datetime.now()))

f = dataset.iloc[:, 3].values
data_range = dataset.shape[0]

# Initialize X and Y
X_amp = np.array([dataset['fft_amp_ref'][i][:NUM_SAMPLES_IN] for i in range(0, data_range)])
Y_amp = np.array([dataset['fft_amp'][i][:NUM_SAMPLES_OUT] for i in range(0, data_range)])
print('X and Y initialized')

alpha = 0.3
activation = 'linear'

model_amp = Sequential()
model_amp.add(Dense(units=NUM_SAMPLES_IN, input_dim=NUM_SAMPLES_IN, kernel_initializer='normal', activation='linear'))
model_amp.add(GaussianNoise(1.0))
#model_amp.add(Dense(units=NUM_SAMPLES_IN, kernel_initializer='normal', activation='linear'))
#model_amp.add(Dense(units=NUM_SAMPLES_IN, kernel_initializer='normal', activation='linear'))
#model_amp.add(Dense(units=NUM_SAMPLES_OUT, kernel_initializer='normal', activation=activation))
model_amp.add(Dense(units=NUM_SAMPLES_OUT, kernel_initializer='normal', activation=activation))
model_amp.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

print('Started at: ' + str(datetime.datetime.now()))
DATA_RANGE = dataset.shape[0]

print('Fit model')
history_amp = model_amp.fit(X_amp, Y_amp, batch_size=34, epochs=50, shuffle=True)

print(history_amp.history.keys())
#  "Accuracy"
plt.subplot(2, 1, 1)
plt.plot(history_amp.history['acc'])
plt.title('Natančnost modela')
plt.ylabel('Natančnost')
plt.xlabel('Ponovitev')
plt.legend(['Trening', 'Validacija'], loc='upper left')

# "Loss"
plt.subplot(2, 1, 2)
plt.plot(history_amp.history['loss'])
plt.title('Napaka modela')
plt.ylabel('Napaka')
plt.xlabel('Ponovitev')
plt.legend(['Trening', 'Validacija'], loc='upper left')
plt.savefig('/home/jangia/Documents/Mag/AudioSystemModel/neural_network/plots/{0}_{1}.png'
            .format('amp_model_loss', datetime.datetime.now()))

plt.close()