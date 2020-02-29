import pyaudio
import numpy as np
import wave
import scipy.signal
import scipy.io.wavfile as wavfile

# user_audio_callback must have signature: callback(in_data)

class AudioDude:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = None

    def audio_callback(self, in_data, frame_count, time_info, status):
        self.user_audio_callback(in_data)
        return (None, pyaudio.paContinue)

    def start_mic_input_stream(self, num_channels, sampling_rate, num_frames_per_buffer, callback):
        self.user_audio_callback = callback
        self.stream = self.audio.open(format=pyaudio.paInt16, channels=num_channels, rate=sampling_rate, input=True, frames_per_buffer=num_frames_per_buffer, stream_callback=self.audio_callback)

    def stop_mic_input_stream(self):
        self.stream.stop_stream()
        self.stream.close()

    def play_wav_file(self, filepath):
        wf = wave.open(filepath, 'rb')
        stream = self.audio.open(format=self.audio.get_format_from_width(wf.getsampwidth()), channels=wf.getnchannels(), rate=wf.getframerate(), output=True)
        data = wf.readframes(1024)
        while len(data):
            stream.write(data)
            data = wf.readframes(1024)

        stream.close()

    def start_mic_output_stream(self, audio_data):
        self.stream.write(audio_data)

    def stop_mic_output_stream(self):
        self.stream.stop_stream()
        self.stream.close()
