# -*- coding: utf-8 -*-
import json

from scipy.fftpack import fft as fft
from scipy.signal import hann
from scipy.stats import signaltonoise as snr
from pymongo import MongoClient
import numpy as np


def process_chunk(filename, frequency, chunk, data_len):

    # database entry
    db_entry = {
        "gain": filename[1],
        "volume": filename[3],
        "frequency": frequency,
        "amp": filename[5:-4].replace('_', '.'),
        "snr": str(snr(chunk)),
        "fft_real": [],
        "fft_imag": []
        }
    fft_real = []
    fft_imag = []

    # calculate FFT
    chunk_fft = fft(chunk * hann(len(chunk)))
    print('Working on file: {0}'.format(filename))
    # add FFTs to db_entry
    for i in range(0, data_len):
        fft_real.append(float(np.real(chunk_fft[i])))
        fft_imag.append(float(np.imag(chunk_fft[i])))

    db_entry['fft_real'] = fft_real
    db_entry['fft_imag'] = fft_imag
      
    # database  
    client = MongoClient()
    db = client.amp
    
    db.fft.insert_one(db_entry)
    print('Finsihed for file: {0}'.format(filename))
    
    return 0


def fft_to_db(wave, samples, frequency, amplitude, data_len):
    
    # database entry
    db_entry = {
        "frequency": str(frequency) + '',
        "amp": str(amplitude) + '',
        "fft_real": [],
        "fft_imag": []
        }

    fft_real = []
    fft_imag = []
    
    # calculate FFT
    chunk_fft = fft(wave[0: samples] * hann(samples))
    
    # add FFTs to db_entry
    for i in range(0, data_len):
        fft_real.append(float(np.real(chunk_fft[i])))
        fft_imag.append(float(np.imag(chunk_fft[i])))

    db_entry['fft_real'] = fft_real
    db_entry['fft_imag'] = fft_imag

    # database
    client = MongoClient()
    db = client.amp
    
    db.fft_ref.insert_one(db_entry)
    
    return 0
