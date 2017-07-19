import os
import numpy as np
from scipy.fftpack import fft, ifft
from scipy.signal import hann
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath('audio_processing.py')))
NUM_SAMPLES_IN = 10000
NUM_SAMPLES_OUT = 10000

model_path = os.path.join(BASE_DIR, 'normal_regression', 'models', 'test_model_random_forest')
model_amp = joblib.load('{path}_{type}.pkl'.format(path=model_path, type='amp'))
model_phi = joblib.load('{path}_{type}.pkl'.format(path=model_path, type='phi'))

FS = 48000
T_END = 1

t = np.arange(FS * T_END)
chunk = 0.9**6 * np.sin(2 * np.pi * 440 * t / FS)
# BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath('audio_processing.py')))
# REC_PATH = os.path.join(BASE_DIR, 'guitar', 'Middle.wav')
#
# rate, chunk = wav.read(os.path.join(BASE_DIR, REC_PATH))
fft_data = fft(chunk[:40000] * hann(40000))[:NUM_SAMPLES_IN]

# fft_data = fft(audio_data[40000:60000])[:10000]

fft_data_amp = np.empty([1, NUM_SAMPLES_IN + 2])
fft_data_amp[0][0] = np.float64(4)
fft_data_amp[0][1] = np.float64(4)
fft_data_amp[0][2:] = np.abs(fft_data)

fft_data_phi = np.empty([1, NUM_SAMPLES_IN + 2])
fft_data_phi[0][0] = np.float64(4)
fft_data_phi[0][1] = np.float64(4)
fft_data_phi[0][2:] = np.unwrap(np.angle(fft_data))

# predicted out
y_pred_phi = np.zeros([1, NUM_SAMPLES_IN]) # model_phi.predict(fft_data_phi)
y_pred_amp = model_amp.predict(fft_data_amp)

y_pred = np.empty([NUM_SAMPLES_OUT], dtype='complex128')

for i in range(0, NUM_SAMPLES_OUT):
    y_pred[i] = y_pred_amp[0][i] * np.cos(y_pred_phi[0][i]) + 1j * y_pred_amp[0][i] * np.sin(y_pred_phi[0][i])

plt.subplot(3, 1, 1)
plt.semilogy(abs(y_pred), 'r')
plt.title('Original vs Modeled')
plt.ylabel('Amplitude')

plt.subplot(3, 1, 2)
plt.plot(ifft(y_pred), 'b')
plt.title('Original vs Modeled')
plt.ylabel('Amplitude')

plt.subplot(3, 1, 3)
plt.plot(chunk, 'g')
plt.title('Original vs Modeled')
plt.ylabel('Amplitude')
plt.show()
