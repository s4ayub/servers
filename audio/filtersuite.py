from scipy.signal import butter, lfilter
import numpy as np

def bpf(data, fs, lowf, highf, order):
    nyq = 0.5 * fs
    lowf /= nyq
    highf /= nyq
    num, den = butter(order, [lowf, highf], btype='band')
    filtered = lfilter(num, den, data)
    return filtered

def generate_sine_data(n, freq, fs):
    x = np.arange(n)
    x = np.sin(2*np.pi*freq*x/fs)
    return x
