# -*- coding: utf-8 -*-
import json

from scipy.fftpack import fft as fft
from scipy.signal import hann
from scipy.stats import signaltonoise as snr
from pymongo import MongoClient
import numpy as np
import matplotlib.pyplot as plt


def process_chunk(filename, frequency, chunk, data_len):

    # database entry
    db_entry = {
        "gain": filename[1],
        "volume": filename[3],
        "frequency": frequency,
        "amp": filename[5:-4].replace('_', '.'),
        "snr": str(snr(chunk)),
        "fft_amp": [],
        "fft_ph": []
        }
    fft_amp = []
    fft_ph = []

    # calculate FFT
    chunk_fft = fft(chunk * hann(len(chunk)))

    # plt.semilogy(abs(chunk_fft[:1500]), 'b')
    # plt.title('Original vs Modeled')
    # plt.ylabel('Amplitude')
    # plt.show()

    # add FFTs to db_entry
    for i in range(0, data_len):
        fft_amp.append(abs(chunk_fft[i]))
        fft_ph.append(np.angle(chunk_fft[i]))

    db_entry['fft_amp'] = fft_amp
    db_entry['fft_ph'] = fft_ph
      
    # database  
    client = MongoClient()
    db = client.amp
    
    db.fft_amp_phi.insert_one(db_entry)
    print('Finsihed for file: {0}'.format(filename))
    
    return 0


def fft_to_db(wave, samples, frequency, amplitude, data_len):
    
    # database entry
    db_entry = {
        "frequency": str(frequency) + '',
        "amp": str(amplitude) + '',
        "fft_amp": [],
        "fft_ph": []
        }

    fft_amp = []
    fft_ph = []
    
    # calculate FFT
    chunk_fft = fft(wave[0: samples] * hann(samples))
    
    # add FFTs to db_entry
    for i in range(0, data_len):
        fft_amp.append(abs(chunk_fft[i]))
        fft_ph.append(np.angle(chunk_fft[i]))

    db_entry['fft_amp'] = fft_amp
    db_entry['fft_ph'] = fft_ph

    # database
    client = MongoClient()
    db = client.amp
    
    db.fft_ref_amp_phi.insert_one(db_entry)
    
    return 0
