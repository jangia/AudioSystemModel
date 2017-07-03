#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 12:25:31 2017

@author: jangia
"""
from scipy.io import wavfile as wav


def cut_record(filepath, step, samples=40000, num_of_chunks=2, offset=0):

    rate, audio = wav.read(filepath)
    audio_data = [(ele/2**16.) for ele in audio]
    chunks = {}
    print('Start to cut to chunks for:{0}'.format(filepath))
    for i in range(0, num_of_chunks - 1):
        start = i * step + offset
        stop = start + samples
        
        frequency = 440 * (2 ** (1 / 12)) ** (i - 30)
        
        chunk = [item[0] for item in audio_data[start:stop]]

        chunks[str(frequency)] = chunk
    print('Finish to cut to chunks for:{0}'.format(filepath))
    return chunks
