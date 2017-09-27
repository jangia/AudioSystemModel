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
del dataset

# first model
activation_1 = 'linear'

model_amp_1 = Sequential()
model_amp_1.add(Dense(units=NUM_SAMPLES_IN, input_dim=NUM_SAMPLES_IN, kernel_initializer='normal', activation=activation_1))
model_amp_1.add(Dense(units=NUM_SAMPLES_IN, kernel_initializer='normal', activation=activation_1))
model_amp_1.add(Dense(units=NUM_SAMPLES_OUT, kernel_initializer='normal', activation=activation_1))
model_amp_1.add(Dense(units=NUM_SAMPLES_OUT, kernel_initializer='normal', activation=activation_1))
model_amp_1.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

print('Fit model1')
history_amp_1 = model_amp_1.fit(X_amp, Y_amp, batch_size=15, epochs=100, shuffle=True)

MODELS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath('config.py'))), config['models_location'])
model_amp_1.save(
    os.path.join(MODELS_PATH, config['neural_network']['models_location'], 'neural_network_small_1_{0}.h5'
                 .format(datetime.datetime.now()))
)
del model_amp_1

# second model
activation_2 = 'linear'

model_amp_2 = Sequential()
model_amp_2.add(Dense(units=NUM_SAMPLES_IN, input_dim=NUM_SAMPLES_IN, kernel_initializer='normal', activation=activation_2))
model_amp_2.add(Dense(units=NUM_SAMPLES_IN, kernel_initializer='normal', activation=activation_2))
model_amp_2.add(Dense(units=NUM_SAMPLES_OUT, kernel_initializer='normal', activation=activation_2))
model_amp_2.add(Dense(units=NUM_SAMPLES_OUT, kernel_initializer='normal', activation=activation_2))
model_amp_2.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

print('Fit model2')
history_amp_2 = model_amp_2.fit(X_amp, Y_amp, batch_size=100, epochs=150, shuffle=True)

MODELS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath('config.py'))), config['models_location'])
model_amp_2.save(
    os.path.join(MODELS_PATH, config['neural_network']['models_location'], 'neural_network_small_2_{0}.h5'
                 .format(datetime.datetime.now()))
)
del model_amp_2

# second model
activation_3 = 'linear'

model_amp_3 = Sequential()
model_amp_3.add(Dense(units=NUM_SAMPLES_IN, input_dim=NUM_SAMPLES_IN, kernel_initializer='normal', activation=activation_3))
model_amp_3.add(Dense(units=NUM_SAMPLES_IN, kernel_initializer='normal', activation=activation_3))
model_amp_3.add(Dense(units=NUM_SAMPLES_OUT, kernel_initializer='normal', activation=activation_3))
model_amp_3.add(Dense(units=NUM_SAMPLES_OUT, kernel_initializer='normal', activation=activation_3))
model_amp_3.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

print('Fit model2')
history_amp_3 = model_amp_3.fit(X_amp, Y_amp, batch_size=500, epochs=150, shuffle=True)

MODELS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath('config.py'))), config['models_location'])
model_amp_3.save(
    os.path.join(MODELS_PATH, config['neural_network']['models_location'], 'neural_network_small_3_{0}.h5'
                 .format(datetime.datetime.now()))
)
del model_amp_3

#  "Accuracy"
plt.subplot(2, 1, 1)
plt.plot(history_amp_1.history['acc'], color='b')
plt.plot(history_amp_1.history['acc'], color='r')
plt.plot(history_amp_3.history['acc'], color='g')
plt.title('Natančnost modela')
plt.ylabel('Natančnost')
plt.xlabel('Ponovitev')
plt.legend(['Podset 15', 'Podset 100', 'Podset 500'], loc='lower right')

# "Loss"
plt.subplot(2, 1, 2)
plt.plot(history_amp_1.history['loss'], color='b')
plt.plot(history_amp_2.history['loss'], color='r')
plt.plot(history_amp_3.history['loss'], color='g')
plt.title('Napaka modela')
plt.ylabel('Napaka')
plt.xlabel('Ponovitev')
plt.legend(['Podset 15', 'Podset 100', 'Podset 500'], loc='upper right')
plt.savefig('/home/jangia/Documents/Mag/AudioSystemModel/neural_network/plots/{0}_{1}.png'
            .format('amp_model_loss', datetime.datetime.now()))

plt.close()