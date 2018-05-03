#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 12:25:31 2017

@author: jangia
"""
import os
import threading

import math
from pymongo import MongoClient
from scipy.fftpack import fft
from scipy.signal import hann
import numpy as np
from scipy.io import wavfile as wav


class AudioWindowFftToDb:

    def __init__(self, subfolder_name, sample_rate=96000, fft_samples=24000, fft_db_len=12000, overlap=20000):
        self.subfolder_name = subfolder_name
        self.sample_rate = sample_rate
        self.fft_samples = fft_samples
        self.fft_db_len = fft_db_len
        self.rec_path = self.set_filepath()
        self.overlap=overlap

    def set_filepath(self):

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath('window_processing.py')))

        return os.path.join(base_dir, 'guitar_recordings', self.subfolder_name)

    def get_files(self):
        ref_path = os.path.join(self.rec_path, 'ref')
        amp_path = os.path.join(self.rec_path, 'amp')

        ref_files = sorted([f for f in os.listdir(ref_path) if os.path.isfile(os.path.join(ref_path, f))])
        amp_files = sorted([f for f in os.listdir(amp_path) if os.path.isfile(os.path.join(amp_path, f))])

        return ref_files, amp_files

    def process_files(self):

        ref_files, amp_files = self.get_files()

        # go through all files
        for i in range(len(ref_files)):

            # get chunks for amp record
            audio_chunks = self.cut_record(
                os.path.join(self.rec_path, 'amp', amp_files[i]),
                self.fft_samples,
                self.overlap
            )

            # get chunks for ref record
            audio_chunks_ref = self.cut_record(
                os.path.join(self.rec_path, 'ref', ref_files[i]),
                self.fft_samples,
                self.overlap
            )

            for j in range(len(audio_chunks)):
                # run proccessing
                #t = threading.Thread(target=self.process_chunk, args=(chunk, audio_chunks_ref, self.fft_db_len))
                #t.daemon = True
                #t.start()
                self.process_chunk(audio_chunks_ref[j], audio_chunks_ref[j], self.fft_db_len)

    def process_chunk(self, chunk, ref_chunk, data_len):

        # database entry
        db_entry = {
            "fft_amp": [],
            "fft_ph": [],
            "fft_amp_ref": [],
            "fft_ph_ref": []
        }
        fft_amp = []
        fft_ph = []
        fft_amp_ref = []
        fft_ph_ref = []

        # calculate FFT
        chunk_fft = fft(chunk)
        chunk_ref_fft = fft(ref_chunk)

        # add FFTs to db_entry
        for i in range(0, data_len):
            fft_amp.append(abs(chunk_fft[i]))
            fft_ph.append(np.angle(chunk_fft[i]))
            fft_amp_ref.append(abs(chunk_ref_fft[i]))
            fft_ph_ref.append(np.angle(chunk_ref_fft[i]))

        db_entry['fft_amp'] = fft_amp
        db_entry['fft_ph'] = fft_ph
        db_entry['fft_amp_ref'] = fft_amp_ref
        db_entry['fft_ph_ref'] = fft_ph_ref

        # database
        client = MongoClient()
        db = client.amp

        db.fft_amp_phi_guitar_solo.insert_one(db_entry)

        return 0

    @staticmethod
    def cut_record(filepath, samples, overlap):
        """
        Cut audio into chunks for FFT
        :return: 
        """
        # Calculate how many chunks from audio
        rate, audio = wav.read(filepath)
        # audio_data = [(ele / 2 ** 16.) for ele in audio]
        audio_data = audio

        num_chunks = int(math.floor(len(audio_data)/(samples - overlap))) - 10
        chunks = []

        for i in range(0, num_chunks - 1):
            start_index = i * (samples - overlap)
            stop_index = start_index + samples

            fft_chunk = audio_data[start_index:stop_index] * hann(samples)

            chunks.append(fft_chunk)

        return chunks


if __name__=='__main__':
    audio_to_db = AudioWindowFftToDb(subfolder_name='IgranjeSolo')
    audio_to_db.process_files()
