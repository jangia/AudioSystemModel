#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 26 17:15:24 2017

@author: jangia
"""
import datetime
import json
import os

import pandas as pd
import numpy as np
from pymongo import MongoClient
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

NUM_SAMPLES = 10000
# create DB connection
client = MongoClient()
db = client.amp

AMPS = [0.90 ** i for i in range(0, 26)]

model_real = Sequential()
model_real.add(Dense(units=NUM_SAMPLES, input_dim=NUM_SAMPLES+2, kernel_initializer='normal'))
model_real.add(Dense(units=NUM_SAMPLES, kernel_initializer='normal'))
model_real.compile(loss='mean_squared_error', optimizer='adam')

model_imag = Sequential()
model_imag.add(Dense(units=NUM_SAMPLES, input_dim=NUM_SAMPLES+2, kernel_initializer='normal'))
model_imag.add(Dense(units=NUM_SAMPLES, kernel_initializer='normal'))
model_imag.compile(loss='mean_squared_error', optimizer='adam')

cnt = 0

for amp in AMPS:

    print('Started at: ' + str(datetime.datetime.now()))

    # Get all FFTs
    fft_ref_all = pd.DataFrame(list(db.fft.find({'amp': str(AMPS[0])})))
    fft_all = pd.DataFrame(list(db.fft.find({'amp': str(AMPS[0])})))

    print('I have data from database at:' + str(datetime.datetime.now()))

    dataset = fft_all.merge(fft_ref_all, how='inner', on=['amp', 'frequency'])

    f = dataset.iloc[:, 3].values

    print('Working for f={f} and amp={amp}'.format(f=str(f[cnt]), amp=amp))

    # Set amplitude and input FFT as input data
    gain = np.float64(dataset.iloc[:, -3].values)[0]
    volume = np.float64(dataset.iloc[:, -1].values)[0]

    X3 = dataset.iloc[:, -4].values

    # Initialize X
    X_re = np.empty([56, NUM_SAMPLES+2])
    X_im = np.empty([56, NUM_SAMPLES+2])

    # Set output FFT
    Y1 = dataset.iloc[:, 2].values

    # Initialize real and imag array of output FFT
    Y_re = np.empty([56, NUM_SAMPLES])
    Y_im = np.empty([56, NUM_SAMPLES])

    print('X and Y initialized')
    print('Filling X and Y with values from database')
    # Convert from string to complex and amplitude
    for i in range(0, len(X3)):

        X_re[i][0] = gain
        X_re[i][1] = volume
        X_im[i][0] = gain
        X_im[i][1] = volume

        for j in range(0, NUM_SAMPLES):

            Y_re[i][j] = np.real(np.char.replace(Y1[i][j], '', '').astype(np.complex128))
            Y_im[i][j] = np.imag(np.char.replace(Y1[i][j], '', '').astype(np.complex128))

            X_re[i][j + 2] = np.real(np.char.replace(X3[i][j], '', '').astype(np.complex128))
            X_im[i][j + 2] = np.imag(np.char.replace(X3[i][j], '', '').astype(np.complex128))

    # Fit model
    print('Fit model')
    model_real.fit(X_re, Y_re, batch_size=56, epochs=12)
    model_imag.fit(X_im, Y_im, batch_size=56, epochs=12)
    #
    # y_pred_im = model_imag.predict(X_im[cnt:cnt+1])
    # y_pred_re = model_real.predict(X_re[cnt:cnt+1])
    #
    # y_pred = np.empty([NUM_SAMPLES], dtype='complex128')
    # y_meas = np.empty([NUM_SAMPLES], dtype='complex128')
    #
    # for i in range(0, NUM_SAMPLES):
    #     y_pred[i] = y_pred_re[0][i] + 1j * y_pred_im[0][i]
    #     y_meas[i] = Y_re[cnt][i] + 1j * Y_im[cnt][i]
    #
    # fig = plt.figure()
    #
    # plt.subplot(2, 1, 1)
    # plt.semilogy(abs(y_meas), 'r')
    # plt.title('Measured Output VS Predicted Output')
    # plt.ylabel('Amplitude')
    #
    # plt.subplot(2, 1, 2)
    # plt.semilogy(abs(y_pred))
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Amplitude')
    #
    # filename = 'g{0}v{1}f{2}a{3}'.format(str(gain), str(volume), str(f[cnt]), str(amp)).replace('.', '_')
    # plt.savefig('/home/jangia/Documents/Mag/MaisterAmpSim/neural_network/plots/{0}.png'.format(filename))
    #
    # plt.close(fig)

    cnt += 1

    if cnt > 24:
        for h in range(0, len(X_im)):
            # Predicting the Test set results
            y_pred_im = model_imag.predict(X_im[h:h + 1])
            y_pred_re = model_real.predict(X_re[h:h + 1])

            y_pred = np.empty([NUM_SAMPLES], dtype='complex128')
            y_meas = np.empty([NUM_SAMPLES], dtype='complex128')

            for i in range(0, NUM_SAMPLES):
                y_pred[i] = y_pred_re[0][i] + 1j * y_pred_im[0][i]
                y_meas[i] = Y_re[h][i] + 1j * Y_im[h][i]

            fig = plt.figure()

            plt.subplot(2, 1, 1)
            plt.semilogy(abs(y_meas), 'r')
            plt.title('Measured Output VS Predicted Output')
            plt.ylabel('Amplitude')

            plt.subplot(2, 1, 2)
            plt.semilogy(abs(y_pred))
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Amplitude')

            filename = 'g{0}v{1}f{2}a{3}'.format(str(gain), str(volume), str(f[h]), str(amp)).replace('.', '_')
            plt.savefig('/home/jangia/Documents/Mag/MaisterAmpSim/neural_network/plots/{0}.png'.format(filename))

            plt.close(fig)

    print('Finished at: ' + str(datetime.datetime.now()))

# serialize model to JSON
BASE_DIR = os.path.dirname(os.path.abspath('test_model.py'))

model_json = model_real.to_json()
with open(os.path.join(BASE_DIR, 'models', 'test_model_real.json'), "w") as json_file:
    json_file.write(json.dumps(json.loads(model_json), indent=4))
model_real.save(os.path.join(BASE_DIR, 'models', 'test_model_real.h5'))

model_json = model_imag.to_json()
with open(os.path.join(BASE_DIR, 'models', 'test_model_imag.json'), "w") as json_file:
    json_file.write(json.dumps(json.loads(model_json), indent=4))
model_imag.save(os.path.join(BASE_DIR, 'models', 'test_model_imag.h5'))
