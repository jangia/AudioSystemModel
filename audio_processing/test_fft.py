import os

import math
from scipy.fftpack import fft, ifft
from scipy.io import wavfile as wav
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hann

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath('audio_processing.py')))
REC_PATH = os.path.join(BASE_DIR, 'guitar', 'Low.wav')
rate, audio_data = wav.read(os.path.join(BASE_DIR, REC_PATH))
FS = 96000
T_END = 0.5
f = 10
t = np.arange(FS * T_END)
#audio_data = 1 * np.sin(2 * np.pi * f * t / FS)

samples_length = 20000
overlap = 10000

num_chunks = int(math.floor(len(audio_data)/(samples_length - overlap)))
chunks = []
b = [(ele/2**8.)*2-1 for ele in audio_data]
new_audio = ifft(fft(b))
print(new_audio.dtype)
all_new_audio = np.empty(200000, dtype='complex128')
print(max(all_new_audio))

# for i in range(0, 15):
#     start_index = i * overlap
#     stop_index = start_index + samples_length
#
#     new_audio = ifft(fft(audio_data[start_index: stop_index] * hann(stop_index-start_index)))
#
#     all_new_audio[start_index:start_index+overlap] += new_audio[0:overlap]
#     all_new_audio[start_index + overlap:start_index + samples_length] = new_audio[overlap:samples_length]
#
#     print(new_audio.dtype)

fig = plt.figure()
plt.subplot(2, 1, 1)
plt.plot(audio_data, 'r')
plt.subplot(2, 1, 2)
plt.plot(np.real(new_audio), 'b')
plt.title('Fft of chunk')
plt.ylabel('Amplitude')


plt.show(fig)
wav.write('/home/jangia/Documents/Mag/MaisterAmpSim/guitar/test.wav', 48000, np.real(new_audio))

