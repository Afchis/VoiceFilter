import librosa


class GetAudio():
    def __init__(self):
        self.sampling_rate = 16000
        self.n_fft = 512
        self.num_mels = 200

    def stft(self, x):
        out = librosa.load(x, sr=self.sampling_rate)
        out = librosa.core.stft(x, n_fft=self.n_fft)
        return out

    def mel(self, x):
        out = librosa.load(x, sr=self.sampling_rate)
        out = librosa.feature.melspectrogram(out, sr=self.sampling_rate,  n_mels=self.num_mels)
        return out