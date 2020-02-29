import AudioDude
import signal
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
import threading
import datetime as dt
import sys
import time
import scipy.signal
import scipy.io.wavfile as wavfile
import argparse
import os
from PIL import Image
from filtersuite import bpf

def print_data(in_data):
    data = np.frombuffer(in_data, dtype=np.int16)
    print(data)

class AudioDudeTester:
    def __init__(self):
        self.ad = AudioDude.AudioDude()
        self.data_lock = threading.Lock()
        self.sampling_frequency = 44100
        self.frames_per_buffer = 1024

        # Graphing
        self.ys = []
        self.fig = plt.figure(frameon=False)
        self.ax = self.fig.add_subplot(1, 1, 1)

        # Recording
        self.input_path = None
        self.output_path = None
        self.timeout = 10
        self.recorded_data = []

        # Filtering
        self.enable_filtering = False
        self.lowf = None
        self.highf = None
        self.f_order = None

    def print_mic_byte_stream(self):
        self.ad.start_mic_input_stream(num_channels=1, sampling_rate=self.sampling_frequency, num_frames_per_buffer=self.frames_per_buffer, callback=print_data)
        signal.pause()

    def graph_mic_byte_stream_animate(self, i, xs, ys):
        if self.ad.stream and len(self.ys):
            self.data_lock.acquire()
            data = self.ys
            self.data_lock.release()
            if self.enable_filtering:
                data = bpf(data, fs=self.sampling_frequency, lowf=self.lowf, highf=self.highf, order=self.f_order)
            self.ax.clear()
            self.ax.plot(data)
            self.ax.set_yticks(range(-255, 255, 500))

    def graph_mic_byte_stream_callback(self, in_data):
        self.data_lock.acquire()
        self.ys = np.frombuffer(in_data, dtype=np.int16).tolist()
        self.data_lock.release()

    def graph_mic_byte_stream(self):
        ani = animation.FuncAnimation(self.fig, self.graph_mic_byte_stream_animate, fargs=(None, self.ys), interval=1)
        self.ad.start_mic_input_stream(num_channels=1, sampling_rate=self.sampling_frequency, num_frames_per_buffer=self.frames_per_buffer, callback=self.graph_mic_byte_stream_callback)
        plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        plt.show()

    def record_mic_byte_stream_callback(self, in_data):
        data = np.frombuffer(in_data, dtype=np.int16).tolist()
        if self.enable_filtering:
            data = bpf(data, fs=self.sampling_frequency, lowf=self.lowf, highf=self.highf, order=self.f_order).tolist()
        self.recorded_data += data

    def record_mic_byte_stream(self):
        timeout = 10
        print("Entered recording mode with filepath='{0}' and timeout={1}s".format(self.output_path, timeout))
        self.ad.start_mic_input_stream(num_channels=1, sampling_rate=self.sampling_frequency, num_frames_per_buffer=self.frames_per_buffer, callback=self.record_mic_byte_stream_callback)
        signal.alarm(timeout)
        input("Press enter to stop recording.\n")
        signal.alarm(0)
        self.ad.stop_mic_input_stream()
        data = np.array(self.recorded_data)
        data = data.astype(np.int16)
        wavfile.write(self.output_path, self.sampling_frequency, data)

    def play_wav_file(self, filepath):
        print("Entered playback mode with filepath='{0}'\n".format(self.input_path))
        self.ad.play_wav_file(filepath)

    def loopback(self):
        filepath = '_loopback.wav'
        print("Entered loopback mode\n")
        while True:
            timeout = 10
            self.ad.start_mic_input_stream(num_channels=1, sampling_rate=self.sampling_frequency, num_frames_per_buffer=self.frames_per_buffer, callback=self.record_mic_byte_stream_callback)
            signal.alarm(timeout)
            input("Recording mode... Press enter for playback mode.")
            signal.alarm(0)
            self.ad.stop_mic_input_stream()
            data = np.array(self.recorded_data)
            data = data.astype(np.int16)
            wavfile.write(filepath, self.sampling_frequency, data)
            self.recorded_data = []

            print("Playback mode...", end='')
            self.ad.play_wav_file(filepath)
            input("Press enter for recording mode")

    def create_spectrogram(self, data, fs, use_logscale=False):
        f, t, Sxx = [None]*3

        if not use_logscale:
            nperseg = 512
            tbins = 1000
            noverlap = (len(data) - tbins*(nperseg))//(-(tbins - 1))
            t_res = (len(data)/fs)*((nperseg-noverlap)/(len(data)-noverlap))
            f_res = fs/(nperseg/2)

            f, t, Sxx = scipy.signal.spectrogram(data, fs, nperseg=nperseg, noverlap=noverlap)
        elif True:
            nperseg = 512
            tbins = 1000
            noverlap = (len(data) - tbins*(nperseg))//(-(tbins - 1))
            t_res = (len(data)/fs)*((nperseg-noverlap)/(len(data)-noverlap))
            f_res = fs/(nperseg/2)
            f, t, Sxx = scipy.signal.spectrogram(data, fs, nperseg=nperseg, noverlap=noverlap)

            fmin = 20
            fmax = 20000
            nf = 1000

            # The following is an excerpt from soundspec: https://github.com/FlorinAndrei/soundspec

            # generate an exponential distribution of frequencies
            # (as opposed to the linear distribution from FFT)
            b = fmin - 1
            a = np.log10(fmax - fmin + 1) / (nf - 1)
            freqs = np.empty(nf, int)
            for i in range(nf):
              freqs[i] = np.power(10, a * i) + b
            # list of frequencies, exponentially distributed:
            freqs = np.unique(freqs)

            # delete frequencies lower than fmin
            fnew = f[f >= fmin]
            cropsize = f.size - fnew.size
            f = fnew
            Sxx = np.delete(Sxx, np.s_[0:cropsize], axis=0)

            # delete frequencies higher than fmax
            fnew = f[f <= fmax]
            cropsize = f.size - fnew.size
            f = fnew
            if cropsize:
                Sxx = Sxx[:-cropsize, :]

            findex = []
            # find FFT frequencies closest to calculated exponential frequency distribution
            for i in range(freqs.size):
              f_ind = (np.abs(f - freqs[i])).argmin()
              findex.append(f_ind)

            # keep only frequencies closest to exponential distribution
            # this is usually a massive cropping of the initial FFT data
            fnew = []
            for i in findex:
              fnew.append(f[i])
            f = np.asarray(fnew)
            Sxxnew = Sxx[findex, :]
            Sxx = Sxxnew
        else:
            # https://www.kaggle.com/himanshurawlani/a-cnn-lstm-model
            window_size = 20
            step_size = 10
            eps = 1e-10
            nperseg = int(round(window_size * fs / 1e3))
            noverlap = int(round(step_size * fs / 1e3))

            f, t, Sxx = scipy.signal.spectrogram(data, fs=fs, window='hann', nperseg=nperseg, noverlap=noverlap, detrend=False)

        return (f, t, Sxx)

    def show_spectrogram(self, filepath, use_logscale=False, use_color=False):
        fs, data = wavfile.read(filepath)
        f, t, Sxx = self.create_spectrogram(data, fs, use_logscale)

        if use_logscale:
            plt.yscale('symlog')
        if use_color:
            plt.pcolormesh(t, f, np.log10(Sxx))
        else:
            plt.pcolormesh(t, f, np.log10(Sxx), cmap=cm.gray)

        plt.ylabel('f [Hz]')
        plt.xlabel('t [sec]')

        plt.show()

    def save_spectrogram_sequence(self, input_folder, output_folder, chunk_size_ms, window_size_ms, window_step_ms, use_logscale=False, use_color=False):
        audio_files = os.listdir(input_folder)
        audio_files = [f for f in audio_files if f.endswith('.wav')]
        bad_chunks = {}
        for f in audio_files:
            print(f)
            input_filename = os.path.basename(f).split('.')[0]
            new_output_folder = os.path.join(output_folder, input_filename)
            if not os.path.exists(new_output_folder):
                os.makedirs(new_output_folder)

            fs, data = wavfile.read(os.path.join(input_folder, f))

            # Segment audio into chunks
            chunk_size = round(chunk_size_ms * (fs/1000))
            chunk_leftover = len(data) % chunk_size
            chunks = [data[x:x+chunk_size] for x in range(0, len(data), chunk_size)]

            # Formulate sliding window
            window_size = round(window_size_ms * (fs/1000))
            window_step = round(window_step_ms * (fs/1000))

            # Create spectrograms using the sliding window
            n = 0
            for chunk in chunks:
                chunk_timestamp = n * chunk_size_ms
                if chunk is chunks[-1] and len(chunk) < chunk_size and len(chunks) > 1:
                    chunk = data[-chunk_size:]
                    chunk_timestamp = round(1000*len(data)/fs) - chunk_size_ms

                output_subfolder = os.path.join(new_output_folder, '%s_%d' % (input_filename, chunk_timestamp))
                if not os.path.exists(output_subfolder):
                    os.makedirs(output_subfolder)

                window_leftover = chunk_size - ((chunk_size - window_size) % window_step)
                m = 0
                for x in range(0, len(chunk), window_step):
                    window_data = chunk[x:x+window_size]
                    output_file = os.path.join(output_subfolder, input_filename)
                    image_path = '%s_%d_%d.png' % (output_file, chunk_timestamp, m * window_step_ms)

                    if len(window_data) < window_size:
                        break

                    f, t, Sxx = self.create_spectrogram(window_data, fs, use_logscale=use_logscale)

                    if not f.any() or not t.any() or not Sxx.any():
                        if output_subfolder not in bad_chunks:
                            print('Bad chunk: %s' % output_subfolder)
                            bad_chunks[output_subfolder] = True

                    if use_color:
                        plt.pcolormesh(t, f, np.log10(Sxx))
                    else:
                        plt.pcolormesh(t, f, np.log10(Sxx), cmap=cm.gray)
                    plt.axis('off')
                    if use_logscale:
                        plt.yscale('symlog')

                    plt.tight_layout(0)
                    plt.gcf().set_dpi(100)
                    plt.gcf().set_figwidth(3.45)
                    plt.gcf().set_figheight(2.57)
                    plt.savefig(image_path, bbox_inches='tight', pad_inches=0, dpi=100)
                    plt.cla()

                    image = Image.open(image_path).convert('L')
                    image.save(image_path)

                    m += 1
                n += 1

        bad_chunks_path = os.path.join(output_folder, 'bad_chunks.txt')
        with open(bad_chunks_path, 'w') as bcf:
            bcf.write('\n'.join(bad_chunks.keys()))

def main():
    mode_choices = ['print','graph','record','play', 'loopback', 'spec', 'nn']
    default_filter_specs = '2000,12000,3'

    custom_formatter = lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=100)
    p = argparse.ArgumentParser(formatter_class=custom_formatter)

    p.add_argument('-m', '--mode', type=str, required=True, choices=mode_choices, metavar="/".join(mode_choices), help="specify mode: " + str(mode_choices))
    p.add_argument('-f', '--filter', type=str, nargs='?', const=default_filter_specs, metavar='LOW,HIGH,ORDER', help="enable bandpass filtering")
    p.add_argument('-i', '--input', type=str, help="specify input path")
    p.add_argument('-o', '--output', type=str, help="specify output path")
    p.add_argument('--use-logscale', action='store_true', help="use logscale for spectrograms")
    p.add_argument('--use-color', action='store_true', help="use color for spectrograms")
    p.add_argument('--nn-prep', action='store_true', help="save spectrograms to disk for neural network training")
    p.add_argument('--chunk-size', type=int, help="nn prep: specify audio chunk size in seconds")
    p.add_argument('--window-size', type=int, help="nn prep: specify sliding window size in milliseconds")
    p.add_argument('--window-step', type=int, help="nn prep: specify sliding window step size in milliseconds")

    args = p.parse_args()
    tester = AudioDudeTester()

    if args.filter:
        filter_specs = list(map(int, args.filter.split(',')))
        if len(filter_specs) is not 3:
            print("Must specify filter specs with LOW,HIGH,ORDER -- e.g. -f 20,20000,3")
            return
        tester.lowf = filter_specs[0]
        tester.highf = filter_specs[1]
        tester.f_order = filter_specs[2]
        tester.enable_filtering = True
        print("\nEnabled bandpass filtering: {0}hz-{1}hz, filter order {2}".format(tester.lowf, tester.highf, tester.f_order))

    if args.input:
       tester.input_path = os.path.expanduser(args.input)
    if args.output:
        tester.output_path = os.path.expanduser(args.output)

    print("")

    if args.mode == 'print':
        tester.print_mic_byte_stream()
    elif args.mode == 'graph':
        tester.graph_mic_byte_stream()
    elif args.mode == 'record':
        if not tester.output_path:
            print("For recording mode, must specify output filepath -- e.g. -m record -o recording.wav")
            return
        tester.record_mic_byte_stream()
    elif args.mode == 'play':
        if not tester.input_path:
            print("For playback mode, must specify input filepath -- e.g. -m play -i recording.wav")
            return
        tester.play_wav_file(tester.input_path) # Implement bandpass filtering for this mode
    elif args.mode == 'loopback':
        tester.loopback()
    elif args.mode == 'spec':
        if not tester.input_path:
            print("For spectrogram mode, must specify input filepath -- e.g. -m spec -i recording.wav")
            return

        use_logscale = True if args.use_logscale else False
        use_color = True if args.use_color else False

        if args.nn_prep:
            if not os.path.isdir(tester.input_path):
                print("For spectrogram NN prep mode, the input path must be a directory")
                return
            if not tester.output_path:
                print("For spectrogram NN prep mode, must specify the output folder to store the sequential histograms")
            if not args.chunk_size or not args.window_size or not args.window_step:
                print("For spectrogram NN prep mode, must specify the chunk size (s), window size (ms), and window step (ms)")
                return

            tester.save_spectrogram_sequence(tester.input_path, tester.output_path, chunk_size_ms=args.chunk_size, window_size_ms=args.window_size, window_step_ms=args.window_step, use_logscale=use_logscale, use_color=use_color)
        else:
            tester.show_spectrogram(tester.input_path, use_logscale=use_logscale, use_color=use_color)

if __name__ == "__main__":
    main()
