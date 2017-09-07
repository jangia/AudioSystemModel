#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 12:25:31 2017

@author: jangia
"""
import os
import threading

from pymongo import MongoClient
from scipy.fftpack import fft
from scipy.signal import hann
import numpy as np
from scipy.io import wavfile as wav


class AudioFftToDb:

    def __init__(self, sample_rate=96000, fft_samples=24000, num_chunks=56, offset=10000, fft_db_len=12000, file_pattern='*'):
        self.sample_rate = sample_rate
        self.fft_samples = fft_samples
        self.num_chunks = num_chunks
        self.offset = offset
        self.fft_db_len = fft_db_len
        self.rec_path = self.set_filepath()
        self.file_pattern = file_pattern

    def set_filepath(self):

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath('processing.py')))

        return os.path.join(base_dir, 'recordings')

    def get_files(self):

        return [f for f in os.listdir(self.rec_path) if os.path.isfile(os.path.join(self.rec_path, f))]

    def process_files(self):

        files = self.get_files()

        for file in files:

            if self.file_pattern in file or self.file_pattern == '*':
                # chunk file
                audio_chunks = self.cut_record(
                    os.path.join(self.rec_path, file),
                    self.sample_rate,
                    self.fft_samples,
                    self.num_chunks,
                    self.offset
                )

                for frq, chunk in audio_chunks.items():
                    # run proccessing
                    t = threading.Thread(target=self.process_chunk, args=(file, frq, chunk, self.fft_db_len))
                    t.daemon = True
                    t.start()
                    #process_chunk(file, frq, chunk, self.fft_db_len)

    def process_chunk(self, filename, frequency, chunk, data_len):

        # database entry
        db_entry = {
            "gain": filename[1],
            "volume": filename[3],
            "frequency": frequency,
            "amp": filename[5:-4].replace('_', '.'),
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

    def cut_record(self, filepath, step, samples=40000, num_of_chunks=2, offset=0):

        rate, audio = wav.read(filepath)
        audio_data = [(ele / 2 ** 16.) for ele in audio]
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


if __name__=='__main__':
    audio_to_db = AudioFftToDb(file_pattern='g4v4')
    audio_to_db.process_files()