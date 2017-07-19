# cut audio into segments for FFT analysis
import math
import os
from threading import Thread
from sklearn.externals import joblib
from scipy.io import wavfile as wav
from scipy.fftpack import fft as fft
from scipy.fftpack import ifft
from scipy.signal import hann
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath('audio_processing.py')))
NUM_SAMPLES_IN = 1200
NUM_SAMPLES_OUT = 1200


class AudioProcessor:

    def __init__(self, audio, samples_length=10000, overlap=5000, gain=4, volume=4, model_name='test_model_random_forest_no_hann'):
        self.audio = [(ele/2**16.) for ele in audio]  # normalize audio
        self.samples_length = samples_length
        self.overlap = overlap
        self.chunks = []
        self.processed_audio = {}
        self.new_chunks = []
        self.model_path = os.path.join(BASE_DIR, 'normal_regression', 'models', model_name)
        self.model_amp = joblib.load('{path}_{type}.pkl'.format(path=self.model_path, type='amp'))
        self.model_phi = joblib.load('{path}_{type}.pkl'.format(path=self.model_path, type='phi'))
        self.gain = gain
        self.volume = volume

    def cut_audio(self):
        """
        Cut audio into chunks for FFT
        :return: 
        """
        # Calculate how many chunks from audio
        num_chunks = int(math.floor(len(self.audio)/(self.samples_length - self.overlap))) - 1
        chunks = []

        for i in range(0, num_chunks):
            start_index = i * self.overlap
            stop_index = start_index + self.samples_length

            # at the end rather take longer array then too short
            if i == num_chunks - 1:
                fft_chunk = self.audio[start_index:] * hann(len(self.audio) - start_index)
            else:
                fft_chunk = self.audio[start_index:stop_index] * hann(stop_index-start_index)

            chunks.append(fft_chunk)

            self.chunks = chunks

        return chunks

    def main_processing(self):
        """
        Main processing of signal, each chunk in new thread
        :return: 
        """
        # First cut audio into chunks
        self.cut_audio()

        # Process each chunk in separated thread
        threads = []

        for key, chunk in enumerate(self.chunks):
            # t = Thread(target=self.process_signal, args=(chunk, key))
            # t.daemon = True
            # threads.append(t)
            # t.start()
            self.process_signal(chunk, key)

        # wait all threads to stop
        for t in threads:
            t.join()

        # Convert processed ffts to audio signal
        self.fft_to_audio()

    def process_signal(self, chunk, key):
        """
        Process signal through model - run it in thread
        :param chunk: 
        :param key: 
        :return: 
        """

        model_amp = self.model_amp
        model_phi = self.model_phi

        FS = 20000
        T_END = 1

        t = np.arange(FS * T_END)
        chunk = 0.9 ** 6 * np.sin(2 * np.pi * 440 * t / FS)
        # BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath('audio_processing.py')))
        # REC_PATH = os.path.join(BASE_DIR, 'guitar', 'Middle.wav')
        #
        # rate, chunk = wav.read(os.path.join(BASE_DIR, REC_PATH))
        fft_data_all = fft(chunk[:20000])
        fft_data = fft_data_all[:1200]
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
        y_pred_phi = model_phi.predict(fft_data_phi)
        y_pred_amp = model_amp.predict(fft_data_amp)

        y_pred = np.empty([NUM_SAMPLES_OUT], dtype='complex128')

        for i in range(0, NUM_SAMPLES_OUT):
            y_pred[i] = y_pred_amp[0][i] * np.cos(y_pred_phi[0][i]) + 1j * y_pred_amp[0][i] * np.sin(y_pred_phi[0][i])

        plt.subplot(2, 1, 1)
        # plt.semilogy(abs(y_pred), 'b')
        # plt.title('Original vs Modeled')
        # plt.ylabel('Amplitude')
        #
        # plt.subplot(2, 1, 2)
        # plt.plot(ifft(y_pred), 'b')
        # plt.title('Original vs Modeled')
        # plt.ylabel('Amplitude')
        # plt.show()

        self.processed_audio[key] = y_pred

    def fft_to_audio(self):
        """
        Create audio from ffts
        :return: 
        """

        new_audio = np.empty(len(self.audio), dtype='complex128')
        num_chunks = int(math.floor(len(self.audio) / (self.samples_length - self.overlap)))

        for j in range(0, len(self.processed_audio)):
            new_chunk_fft = (self.processed_audio[j])
            # fig = plt.figure()
            # plt.semilogy(new_chunk_fft, 'r')
            # plt.title('Fft of chunk')
            # plt.ylabel('Amplitude')
            # plt.show()
            start_index = j * self.overlap
            stop_index = start_index + self.samples_length

            # plt.close(fig)
            new_audio_chunk = np.fft.ifft(new_chunk_fft)
            # add chunk to whole audio
            if j == num_chunks - 1:
                new_audio[start_index:start_index+self.overlap] += new_audio_chunk[:self.overlap]
                new_audio[start_index + self.overlap:] += new_audio_chunk[self.overlap:]
            else:
                new_audio[start_index:start_index+self.overlap] += new_audio_chunk[:self.overlap]
                new_audio[start_index + self.overlap:stop_index] += new_audio_chunk[self.overlap:]
        fig = plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(new_audio[:50000], 'r')
        plt.title('Fft of chunk')
        plt.ylabel('Amplitude')

        plt.subplot(2, 1, 2)
        plt.plot(self.audio[:50000], 'b')
        filename = 'signal_whole'
        plt.savefig('/home/jangia/Documents/Mag/AudioSystemModel/audio_processing/plots/{0}.png'.format(filename))

        plt.close(fig)
        wav.write('/home/jangia/Documents/Mag/AudioSystemModel/guitar/test.wav', 48000, np.real(new_audio))


if __name__== '__main__':
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath('audio_processing.py')))
    REC_PATH = os.path.join(BASE_DIR, 'guitar', 'Middle.wav')

    rate, audio_data = wav.read(os.path.join(BASE_DIR, REC_PATH))

    audio_processor = AudioProcessor(audio_data, samples_length=1200, overlap=0)
    audio_processor.main_processing()
