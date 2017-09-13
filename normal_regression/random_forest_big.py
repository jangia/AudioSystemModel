# random forest for 12000 samples
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
from sklearn.ensemble import RandomForestRegressor

from config import load_config

config = load_config()
NUM_SAMPLES_IN = config['big_models']['num_samples_in']
NUM_SAMPLES_OUT = config['big_models']['num_samples_out']
NUM_ESTIMATORS = config['random_forest']['big_model']['num_of_estimators']
NUM_JOBS = config['random_forest']['big_model']['num_of_jobs']

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

regressor_amp = RandomForestRegressor(n_estimators=NUM_ESTIMATORS, verbose=3, n_jobs=NUM_JOBS)
print('Fit model at: ' + str(datetime.datetime.now()))
regressor_amp.fit(X_amp, Y_amp)
print('Model fitted at: ' + str(datetime.datetime.now()))

# save model
MODELS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath('config.py'))), config['models_location'])
joblib.dump(
    regressor_amp,
    os.path.join(MODELS_PATH, config['random_forest']['models_location'], 'random_forest_big_{0}.pkl'.format(datetime.datetime.now()))
)

print('Finished at: ' + str(datetime.datetime.now()))
