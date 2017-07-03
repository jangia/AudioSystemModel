# cut audio into segments for FFT analysis
import math
import os
from threading import Thread

from keras.models import load_model
from scipy.io import wavfile as wav
from scipy.fftpack import fft as fft
from scipy.fftpack import ifft
from scipy.signal import hann
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath('audio_processing.py')))


class AudioProcessor:

    def __init__(self, audio, samples_length=10000, overlap=5000, gain=4, volume=4, model_name='test_model'):
        self.audio = [(ele/2**16.) for ele in audio]  # normalize audio
        self.samples_length = samples_length
        self.overlap = overlap
        self.chunks = []
        self.processed_audio = {}
        self.new_chunks = []
        self.model_path = os.path.join(BASE_DIR, 'neural_network', 'models', model_name)
        self.model_amp = self.load_model('amp')
        self.model_phi = self.load_model('phi')
        self.gain = gain
        self.volume = volume

    def load_model(self, type):
        # load weights into new model
        loaded_model = load_model('{path}_{type}.h5'.format(path=self.model_path, type=type))
        print("Loaded model from disk")

        return loaded_model

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
        FS = 48000
        T_END = 1

        f = 440 * (2 ** (1 / 12)) ** 0
        t = np.arange(FS * T_END)
        chunk = 1 * np.sin(2 * np.pi * f * t / FS)
        fft_data = fft(chunk * hann((len(chunk))))[:10000]

        fft_data_amp = np.empty([1, 10002])
        fft_data_amp[0][0] = np.float64(self.gain)
        fft_data_amp[0][1] = np.float64(self.volume)
        fft_data_amp[0][2:] = np.abs(fft_data)
        fft_pred_amp = self.model_amp.predict(fft_data_amp)

        fft_data_phi = np.empty([1, 10002])
        fft_data_phi[0][0] = np.float64(self.gain)
        fft_data_phi[0][1] = np.float64(self.volume)
        fft_data_phi[0][2:] = np.angle(fft_data)
        fft_pred_phi = self.model_phi.predict(fft_data_phi)

        fft_pred = np.empty([10000], dtype='complex128')
        for i in range(0, len(fft_pred_amp)):
            fft_pred[i] = fft_pred_amp[0][i] * np.cos(fft_pred_phi[0][i]) + 1j * fft_pred_amp[0][i] * np.sin(fft_pred_phi[0][i])

        new_audio = np.fft.ifft(chunk)
        # wav.write('/home/jangia/Documents/Mag/AudioSytemModel/guitar/test{0}.wav'.format(key), 48000, chunk)
        wav.write('/home/jangia/Documents/Mag/AudioSystemModel/guitar/test{0}.wav'.format(key), 48000, np.real(new_audio))

        # plt.subplot(2, 1, 1)
        # plt.plot(new_audio, 'r')
        # plt.title('Measured Output VS Predicted Output')
        # plt.ylabel('Amplitude')
        #
        # plt.subplot(2, 1, 2)
        # plt.plot(chunk, 'b')
        # plt.xlabel('Frequency (Hz)')
        # plt.ylabel('Amplitude')
        #
        # plt.show()

        plt.subplot(2, 1, 1)
        plt.semilogy(abs(fft_data[:1500]), 'r')
        plt.title('Measured Output VS Predicted Output')
        plt.ylabel('Amplitude')

        plt.subplot(2, 1, 2)
        plt.semilogy(abs(fft_pred[:1500]))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')

        plt.show()
        self.processed_audio[key] = fft_pred

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
        plt.plot(new_audio, 'r')
        plt.title('Fft of chunk')
        plt.ylabel('Amplitude')

        plt.subplot(2, 1, 2)
        plt.plot(self.audio, 'b')
        filename = 'signal_whole'
        plt.savefig('/home/jangia/Documents/Mag/AudioSystemModel/audio_processing/plots/{0}.png'.format(filename))

        plt.close(fig)
        wav.write('/home/jangia/Documents/Mag/AudioSystemModel/guitar/test.wav', 48000, np.real(new_audio))
        print(max(np.real(new_audio)))

if __name__== '__main__':
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath('audio_processing.py')))
    REC_PATH = os.path.join(BASE_DIR, 'guitar', 'sine.wav')

    rate, audio_data = wav.read(os.path.join(BASE_DIR, REC_PATH))

    audio_processor = AudioProcessor(audio_data)
    audio_processor.main_processing()
