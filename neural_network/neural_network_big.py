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
NUM_SAMPLES_IN = config['big_models']['num_samples_in']
NUM_SAMPLES_OUT = config['big_models']['num_samples_out']

# create DB connection
client = MongoClient()
collection = client.amp.time_domain

print('Started at: ' + str(datetime.datetime.now()))

# Get all FFTs
dataset = pd.DataFrame(list(collection.find({})))
print('I have data from database at:' + str(datetime.datetime.now()))

data_range = dataset.shape[0]

# Initialize X and Y
X_amp = np.array([dataset['in'][i][:NUM_SAMPLES_IN] for i in range(0, data_range)])
Y_amp = np.array([dataset['out'][i][:NUM_SAMPLES_OUT] for i in range(0, data_range)])
print('X and Y initialized')
del dataset

# first model
activation_1 = 'linear'

model_amp_1 = Sequential()
model_amp_1.add(Dense(units=NUM_SAMPLES_IN, input_dim=NUM_SAMPLES_IN, kernel_initializer='normal', activation=activation_1))
model_amp_1.add(Dense(units=NUM_SAMPLES_IN, input_dim=NUM_SAMPLES_IN, kernel_initializer='normal', activation=activation_1))
model_amp_1.add(Dense(units=NUM_SAMPLES_IN, input_dim=NUM_SAMPLES_IN, kernel_initializer='normal', activation=activation_1))
model_amp_1.add(Dense(units=NUM_SAMPLES_IN, input_dim=NUM_SAMPLES_IN, kernel_initializer='normal', activation=activation_1))
model_amp_1.add(Dense(units=NUM_SAMPLES_IN, input_dim=NUM_SAMPLES_IN, kernel_initializer='normal', activation=activation_1))
model_amp_1.add(Dense(units=NUM_SAMPLES_IN, input_dim=NUM_SAMPLES_IN, kernel_initializer='normal', activation=activation_1))
model_amp_1.add(Dense(units=NUM_SAMPLES_IN, input_dim=NUM_SAMPLES_IN, kernel_initializer='normal', activation=activation_1))
model_amp_1.add(Dense(units=NUM_SAMPLES_IN, input_dim=NUM_SAMPLES_IN, kernel_initializer='normal', activation=activation_1))
model_amp_1.add(Dense(units=NUM_SAMPLES_IN, input_dim=NUM_SAMPLES_IN, kernel_initializer='normal', activation=activation_1))
model_amp_1.add(Dense(units=NUM_SAMPLES_IN, input_dim=NUM_SAMPLES_IN, kernel_initializer='normal', activation=activation_1))
model_amp_1.add(Dense(units=NUM_SAMPLES_IN, input_dim=NUM_SAMPLES_IN, kernel_initializer='normal', activation=activation_1))
model_amp_1.add(Dense(units=NUM_SAMPLES_IN, input_dim=NUM_SAMPLES_IN, kernel_initializer='normal', activation=activation_1))
model_amp_1.add(Dense(units=NUM_SAMPLES_IN, input_dim=NUM_SAMPLES_IN, kernel_initializer='normal', activation=activation_1))
model_amp_1.add(Dense(units=NUM_SAMPLES_IN, input_dim=NUM_SAMPLES_IN, kernel_initializer='normal', activation=activation_1))
model_amp_1.add(Dense(units=NUM_SAMPLES_IN, input_dim=NUM_SAMPLES_IN, kernel_initializer='normal', activation=activation_1))
model_amp_1.add(Dense(units=NUM_SAMPLES_IN, input_dim=NUM_SAMPLES_IN, kernel_initializer='normal', activation=activation_1))
model_amp_1.add(Dense(units=NUM_SAMPLES_IN, input_dim=NUM_SAMPLES_IN, kernel_initializer='normal', activation=activation_1))
model_amp_1.add(Dense(units=NUM_SAMPLES_IN, input_dim=NUM_SAMPLES_IN, kernel_initializer='normal', activation=activation_1))
model_amp_1.add(Dense(units=NUM_SAMPLES_IN, input_dim=NUM_SAMPLES_IN, kernel_initializer='normal', activation=activation_1))
model_amp_1.add(Dense(units=NUM_SAMPLES_IN, input_dim=NUM_SAMPLES_IN, kernel_initializer='normal', activation=activation_1))
model_amp_1.add(Dense(units=NUM_SAMPLES_IN, input_dim=NUM_SAMPLES_IN, kernel_initializer='normal', activation=activation_1))
# model_amp_1.add(GaussianNoise(1.0))
model_amp_1.add(Dense(units=NUM_SAMPLES_OUT, kernel_initializer='normal', activation='sigmoid'))
model_amp_1.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

print('Fit model1')
history_amp_1 = model_amp_1.fit(X_amp, Y_amp, batch_size=50, epochs=250, shuffle=True)

MODELS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath('config.py'))), config['models_location'])
model_amp_1.save(
    os.path.join(MODELS_PATH, config['neural_network']['models_location'], 'neural_network_small_1_{0}.h5'
                 .format(datetime.datetime.now()))
)

#  "Accuracy"
plt.subplot(2, 1, 1)
plt.plot(history_amp_1.history['acc'], color='b')
plt.title('Natančnost modela')
plt.ylabel('Natančnost')
plt.xlabel('Ponovitev')

# "Loss"
plt.subplot(2, 1, 2)
plt.plot(history_amp_1.history['loss'], color='b')
plt.title('Napaka modela')
plt.ylabel('Napaka')
plt.xlabel('Ponovitev')
plt.savefig('/home/jangia/Documents/Mag/AudioSystemModel/neural_network/plots/{0}_{1}.png'
            .format('amp_model_loss', datetime.datetime.now()))

plt.close()