import pyaudio
import wave


class Microphone:
    def __init__(self):
        self.FORMAT = pyaudio.paInt16
        self.CHANNELS = 2
        self.RATE = 44100
        self.CHUNK = 1024
        self.RECORD_SECONDS = 5
        self.WAVE_OUTPUT_FILENAME = "./recording/file.wav"
        self.frames = []

        self.audio = pyaudio.PyAudio()

        self.stream = self.audio.open(format=self.FORMAT, channels=self.CHANNELS, rate=self.RATE, input=True,
                                      frames_per_buffer=self.CHUNK)


class Recording(Microphone):
    def __init__(self):
        super().__init__()

    def start_recording(self):
        print("Recording...")

        for i in range(0, int(self.RATE / self.CHUNK * self.RECORD_SECONDS)):
            data = self.stream.read(self.CHUNK)
            self.frames.append(data)

        print("Finished recording!")

    def stop_recording(self):
        self.stream.stop_stream()
        self.stream.close()
        self.audio.terminate()

        waveFile = wave.open(self.WAVE_OUTPUT_FILENAME, 'wb')
        waveFile.setnchannels(self.CHANNELS)
        waveFile.setsampwidth(self.audio.get_sample_size(self.FORMAT))
        waveFile.setframerate(self.RATE)
        waveFile.writeframes(b''.join(self.frames))
        waveFile.close()
