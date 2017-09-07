#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 12:25:31 2017

@author: jangia
"""

import threading
import numpy as np
from scipy.fftpack import fft as fft
from scipy.signal import hann
from pymongo import MongoClient
import numpy as np


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


AMPS = [0.90**i for i in range(0, 26)]
FS = 96000
F0 = 10
F1 = 2000
T_END = 1
D_TYPE = 'float32'

for amp in AMPS:

    samples = np.zeros(shape=(1))
    
    for i in range(-30, 8):
                f = 440 * (2 ** (1 / 12)) ** i
                t = np.arange(FS * T_END)
                sine_wave = amp * np.sin(2 * np.pi * f * t / FS)
                
                t = threading.Thread(target=fft_to_db, args=(sine_wave, 24000, f, amp, 12000))
                t.daemon = True
                t.start()

