#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
from sklearn.ensemble import RandomForestRegressor

NUM_SAMPLES_IN = 1200
NUM_SAMPLES_OUT = 1200
LEVEL_DROP = 600

DATA_RANGE = 2886
# create DB connection
client = MongoClient()
db = client.amp

print('Started at: ' + str(datetime.datetime.now()))

# Get all FFTs
fft_ref_all = pd.DataFrame(list(db.fft_ref.find({})))
fft_all = pd.DataFrame(list(db.fft.find({})))

print('I have data from database at:' + str(datetime.datetime.now()))

dataset = pd.merge(fft_all, fft_ref_all, how='inner', on=['amp', 'frequency'])
dataset = dataset.sample(frac=1).reset_index(drop=True)

f = dataset.iloc[:, 3].values

# Set amplitude and input FFT as input data
gain = dataset.iloc[:, -6].values.astype('float64')
volume = dataset.iloc[:, -4].values.astype('float64')

# Initialize X
X_raw_re = dataset.iloc[:, -1].values
X_raw_im = dataset.iloc[:, -2].values

# Initialize X
X_re = np.empty([DATA_RANGE, NUM_SAMPLES_IN+2])
X_im = np.empty([DATA_RANGE, NUM_SAMPLES_IN+2])

# Set output FFT
Y_raw_re = dataset.iloc[:, 3].values
Y_raw_im = dataset.iloc[:, 2].values

# Initialize real and imag array of output FFT
Y_re = np.empty([DATA_RANGE, NUM_SAMPLES_OUT])
Y_im = np.empty([DATA_RANGE, NUM_SAMPLES_OUT])

print('X and Y initialized')
print('Filling X and Y with values from database')
# Convert from string to complex and amplitude
for i in range(0, DATA_RANGE):

    X_re[i][0] = gain[i]
    X_re[i][1] = volume[i]
    X_im[i][0] = gain[i]
    X_im[i][1] = volume[i]

    for j in range(0, max(NUM_SAMPLES_OUT, NUM_SAMPLES_IN)):

        if j < NUM_SAMPLES_OUT:
            Y_re[i][j] = Y_raw_re[i][j]
            Y_im[i][j] = Y_raw_im[i][j]

        if j < NUM_SAMPLES_IN:
            X_re[i][j + 2] = X_raw_re[i][j]
            X_im[i][j + 2] = X_raw_im[i][j]


print('X and Y initialized')

print('Fit model')
regressor_re = RandomForestRegressor(n_estimators=300)
regressor_re.fit(X_re, Y_re)

regressor_im = RandomForestRegressor(n_estimators=300)
regressor_im.fit(X_im, Y_im)