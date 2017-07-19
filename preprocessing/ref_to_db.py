#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 12:25:31 2017

@author: jangia
"""

import threading
import numpy as np
from preprocessing.process_record.process_chunk import fft_to_db


AMPS = [0.90**i for i in range(0, 26)]
FS = 20000
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
                
                t = threading.Thread(target=fft_to_db, args=(sine_wave, 20000, f, amp, 15000))
                t.daemon = True
                t.start()

