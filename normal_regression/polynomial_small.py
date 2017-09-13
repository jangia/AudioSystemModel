# random forest for 1200 samples
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
from sklearn.preprocessing import PolynomialFeatures

from config import load_config

config = load_config()
NUM_SAMPLES_IN = config['big_models']['num_samples_in']
NUM_SAMPLES_OUT = config['big_models']['num_samples_out']
DEGREE = config['polynomial']['small_model']['degree']

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
f = dataset.iloc[:, 3].values
data_range = dataset.shape[0]

# Initialize X and Y
X_amp = np.array([dataset['fft_amp_x'][i][:NUM_SAMPLES_IN] for i in range(0, data_range)])
Y_amp = np.array([dataset['fft_amp_y'][i][:NUM_SAMPLES_OUT] for i in range(0, data_range)])
print('X and Y initialized')

regressor_amp = PolynomialFeatures(degree=DEGREE)
X_poly = regressor_amp.fit_transform(X_amp)
print('Fit model at: ' + str(datetime.datetime.now()))
regressor_amp.fit(X_poly, Y_amp)
print('Model fitted at: ' + str(datetime.datetime.now()))

# save model
MODELS_PATH = os.path.join(os.path.dirname(os.path.abspath('config.py')), config['models_location'])
joblib.dump(
    regressor_amp,
    os.path.join(MODELS_PATH, config['polynomial']['models_location'], 'polynomial_big_{0}.pkl'.format(datetime.datetime.now()))
)

print('Finished at: ' + str(datetime.datetime.now()))
