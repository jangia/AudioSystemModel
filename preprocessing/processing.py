#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 25 12:25:31 2017

@author: jangia
"""
import os
import threading
from preprocessing.process_record.edit_record import cut_record
from preprocessing.process_record.process_chunk import process_chunk


class AudioFftToDb:

    def __init__(self, sample_rate=96000, fft_samples=96000, num_chunks=38, offset=10000, fft_db_len=20000, file_pattern='*'):
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
                audio_chunks = cut_record(
                    os.path.join(self.rec_path, file),
                    self.sample_rate,
                    self.fft_samples,
                    self.num_chunks,
                    self.offset
                )

                for frq, chunk in audio_chunks.items():
                    # run proccessing
                    t = threading.Thread(target=process_chunk, args=(file, frq, chunk, 15000))
                    t.daemon = True
                    t.start()


if __name__=='__main__':
    audio_to_db = AudioFftToDb(file_pattern='*')
    audio_to_db.process_files()