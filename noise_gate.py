from scipy.io import wavfile
from scipy.fft import fft, ifft

import matplotlib.pyplot as plt
import numpy as np

import sys

import pandas as pd


def generate_audio(freqs, freq_strengths, sampling_rate=44100, length=1.0):
    # length is length of clip in seconds, freqs is frequencies present in Hz, freq_strengths is strength of frequencies

    final_audio = [0]*int(sampling_rate*length)
    for i in range(len(freqs)):
        freq = freqs[i]
        strength = freq_strengths[i]
        for j in range(int(sampling_rate*length)):
            final_audio[j] += strength*np.sin(freq*2*np.pi*(j/sampling_rate))

    return final_audio


if __name__ == "__main__":
    input_filename = input("Please input filename here: ")
    if input_filename[-3:] != 'wav':
        print('WARNING!! Input File format should be *.wav')
        sys.exit()

    sampling_rate, data = wavfile.read(input_filename)

    data = pd.DataFrame(data)
    data = data.drop(range(44100, len(data)), axis=0)
    data = data.drop(1, axis=1)
    data.columns = ["mono"]

    data["theory"] = generate_audio(freqs=[440], freq_strengths=[2000], length=1.0)
    data["fft"] = fft(np.array(data["mono"]))

    alt_fft = []
    max_fourier_coeff = max(abs(data["fft"]))*0.4
    for x in data["fft"]:
        if abs(x) > max_fourier_coeff:
            alt_fft.append(x)
        else:
            alt_fft.append(0)

    data["alt fft"] = alt_fft
    data["alt mono"] = ifft(np.array(data["alt fft"]))

    data["final mono"] = np.array(data["alt mono"]).real

    plt.plot(abs(data["fft"]))
    plt.ylabel("Strength of frequency in recording")
    plt.xlabel("Frequency")
    plt.show()

    plt.plot(data["mono"][0:1000])
    plt.plot(data["final mono"][0:1000])
    plt.plot(data["theory"][0:1000])
    plt.legend(["original audio", "cleaned audio", "theoretical audio"])
    plt.xlabel("time")
    plt.show()