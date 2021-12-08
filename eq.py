from scipy.io import wavfile
from scipy.fft import fft, ifft

import matplotlib.pyplot as plt
import numpy as np

import sys

import pandas as pd

def eq(fourier_coeffs, subbass, bass, low_mids, high_mids, presence, brilliance):
    alt_fourier_coeffs = []

    n = len(fourier_coeffs)
    for i in range(n):
        fourier_coeff = fourier_coeffs[i]
        if i < 16 or n-i < 16: alt_fourier_coeffs.append(0)
        elif 16 <= i < 60 or 16 <= n-i < 60: alt_fourier_coeffs.append(fourier_coeff*subbass)
        elif 60 <= i < 250 or 60 <=  n-i < 250: alt_fourier_coeffs.append(fourier_coeff*bass)
        elif 250 <= i < 2000 or 250 <= n-i < 2000: alt_fourier_coeffs.append(fourier_coeff*low_mids)
        elif 2000 <= i < 4000 or 2000 <= n-i < 4000: alt_fourier_coeffs.append(fourier_coeff*high_mids)
        elif 4000 <= i < 6000 or 4000 <= n-i < 6000: alt_fourier_coeffs.append(fourier_coeff*presence)
        elif 6000 <= i < 16000 or 6000 <= n-i < 16000: alt_fourier_coeffs.append(fourier_coeff*brilliance)
        else: alt_fourier_coeffs.append(0)

    return alt_fourier_coeffs


if __name__ == "__main__":
    input_filename = input("Please input filename here: ")

    if input_filename[-3:] != 'wav':
        print('WARNING!! Input File format should be *.wav')
        sys.exit()

    sampling_rate, data = wavfile.read(input_filename)

    data = pd.DataFrame(data)
    data = data.drop(range(44100, len(data)), axis=0)

    data.columns = ["mono"]

    data["fft"] = fft(np.array(data["mono"]))
    data["alt fft"] = eq(data["fft"], 0, 0, 1, 0, 0, 0)

    plt.plot(abs(data["fft"]))
    plt.plot(abs(data["alt fft"]))
    plt.legend(["Original DFT", "Altered DFT"])
    plt.ylabel("Relative presence of frequency in recording")
    plt.xlabel("Frequency (Hz)")

    n = len(data)
    bands = [16, 60, 250, 2000, 4000, 6000, 16000]
    for band in bands:
        plt.axvline(band, ls="-", lw=1, c="lightgrey")
        plt.axvline(n-band, ls="-", lw=1, c="lightgrey")

    plt.show()

    data["alt mono"] = ifft(np.array(data["alt fft"]))
    data["final mono"] = np.array(data["alt mono"]).real

    plt.plot(data["mono"][0:1000])
    plt.plot(data["final mono"][0:1000])
    plt.legend(["Original Audio", "Cleaned Audio"])
    plt.xlabel("Time")
    plt.ylabel("Relative air pressure")
    plt.show()