import pyaudio
import wave
import struct
import numpy as np
import time


class Recorder:
    def __init__(self, chunk=1024, silence_threshold=50, silence_timeout=3, channels=2, fs=16000, swidth=2):
        # Record in chunks of 1024 samples
        self.chunk = chunk
        self.silence_threshold = silence_threshold
        self.silence_timeout = silence_timeout
        # 16 bits per sample
        self.sample_format = pyaudio.paInt16
        self.channels = channels
        # Record at 16000 samples per second
        self.fs = fs
        self.swidth = swidth
        # Create an interface to PortAudio
        self.p = pyaudio.PyAudio()
        self.stream = self.p.open(
            format=self.sample_format,
            channels=self.channels,
            rate=self.fs,
            frames_per_buffer=self.chunk,
            input=True
        )
        self.talking = False
        self.finished_rec = False

    def rms(self, frame):
        count = len(frame) / self.swidth
        format = "%dh" % (count)
        shorts = struct.unpack(format, frame)
        # Normalize the frame data by 1/2^15 = 1/32768 since the format is int16
        sum_squares = np.sum(np.power(np.int64(shorts), 2)) * 1/32768 ** 2
        rms = np.power(sum_squares / count, 0.5)
        return rms * 1000

    def record_until_silence(self):
        # Initialize array to store frames
        frames = []
        print("Recording starts first instance of noise")
        self.finished_rec = False
        end = time.time() + self.silence_timeout
        # Store data in chunks
        while time.time() < end:
            data = self.stream.read(self.chunk)
            # Extend recording time if the rms is above the threshold
            if self.rms(data) > self.silence_threshold:
                end = time.time() + self.silence_timeout
                if not self.talking:
                    print("Talking detected, recording started")
                    self.talking = True
                    audio_recording = True  
            elif not self.talking:
                end = time.time() + self.silence_timeout
            if self.talking:
                frames.append(data)
        print("Ending recording")
        self.talking = False
        return frames

    def stop_recording(self):
        # Stop and close the stream
        self.stream.stop_stream()
        self.stream.close()
        # Terminate the PortAudio interface
        self.p.terminate()
        self.finished_rec = True
        print('Finished recording at ', time.time(), ' seconds')

    def save_recording(self, frames, filename="output.wav"):
        # Save the recorded data as a WAV file
        wf = wave.open(filename, 'wb')
        wf.setnchannels(self.channels)
        wf.setsampwidth(self.p.get_sample_size(self.sample_format))
        wf.setframerate(self.fs)
        wf.writeframes(b''.join(frames))
        wf.close()

