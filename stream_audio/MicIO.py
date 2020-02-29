import pyaudio
import numpy as np
import wave
import scipy.signal
import scipy.io.wavfile as wavfile

class MicIO:
    NumberOfChannels = 1
    SamplingFrequency = 44100
    FramesPerBuffer = 1024

    def __init__(self):
        self.pyaudio_io = pyaudio.PyAudio()
        self.pyaudio_stream = None
        self.user_callback = None

    def callback(self, in_data, frame_count, time_info, status):
        self.user_callback(in_data)
        return (None, pyaudio.paContinue)

    def listen(self, callback, device_index=None):
        self.user_callback = callback
        self.pyaudio_stream = self.pyaudio_io.open(format=pyaudio.paInt16,
                                                   channels=MicIO.NumberOfChannels,
                                                   rate=MicIO.SamplingFrequency,
                                                   input=True,
                                                   input_device_index=device_index,
                                                   frames_per_buffer=MicIO.FramesPerBuffer,
                                                   stream_callback=self.callback)

    def stop(self):
        self.pyaudio_stream.stop_stream()
        self.pyaudio_stream.close()

    def play(self, wavfile, device_index=None):
        wf = wave.open(wavfile, 'rb')
        stream = self.pyaudio_stream.open(format=self.audio.get_format_from_width(wf.getsampwidth()),
                                          channels=wf.getnchannels(),
                                          rate=wf.getframerate(),
                                          output=True,
                                          output_device_index=device_index)

        data = wf.readframes(MicIO.FramesPerBuffer)
        while len(data):
            stream.write(data)
            data = wf.readframes(MicIO.FramesPerBuffer)

        stream.close()
