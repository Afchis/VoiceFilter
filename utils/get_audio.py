import os, glob, random

import librosa
import numpy as np


class GetAudio():
    def __init__(self, data_path, data_format="*.flac"):
        self.data_path = data_path
        self.data_format = data_format
        self.sampling_rate = 16000
        self.n_fft = 1200
        self.num_mels = 512
        self.mels_list = [100, 120, 130, 150]
        self.mel_basis = librosa.filters.mel(sr=self.sampling_rate,
                                             n_fft=512,
                                             n_mels=40)

    def _mix_wave(self, x1, x2):
        if x1.shape > x2.shape:
            time_dif = x1.shape[0] - x2.shape[0]
            time = random.randrange(time_dif)
            x2 = np.hstack([np.zeros(time), x2, np.zeros(time_dif-time)])
            return x1, x1 + x2
        else:
            time_dif = x2.shape[0] - x1.shape[0]
            time = random.randrange(time_dif)
            x1 = np.hstack([np.zeros(time), x1, np.zeros(time_dif-time)])
            return x1, x1 + x2

    def _mel_encoder(self, x):
        spec = librosa.core.stft(x, n_fft=512)
        magnitudes = np.abs(spec) ** 2
        mel = np.log10(np.dot(self.mel_basis, magnitudes) + 1e-6)
        return mel
        
    def _stft(self, x):
        return librosa.core.stft(x, n_fft=self.n_fft)

    def _mel(self, x):
        return librosa.feature.melspectrogram(x, sr=self.sampling_rate,  n_mels=self.num_mels)

    def _mel_stack(self, x):
        mels = list()
        for i in range(self.mels_list):
            mels.append(librosa.feature.melspectrogram(x, sr=self.sampling_rate,  n_mels=self.mels_list[i]))
        out = np.concatenate(mels)
        return out

    def train_data(self):
        _, id_names, _ = next(os.walk(self.data_path))
        id_name = random.sample(id_names, 2)
        target_files = random.sample(glob.glob(os.path.join(self.data_path, id_name[0], self.data_format)), 2)
        inter_file = random.choice(glob.glob(os.path.join(self.data_path, id_name[1], self.data_format)))
        refer_wave, _ = librosa.load(target_files[0], sr=self.sampling_rate)
        clear_wave, _ = librosa.load(target_files[1], sr=self.sampling_rate)
        inter_wave, _ = librosa.load(inter_file, sr=self.sampling_rate)
        clear_wave, noicy_wave = self._mix_wave(clear_wave, inter_wave)
        # refer_spec = self._stft(refer_wave)
        refer_spec = self._mel_encoder(refer_wave)
        clear_spec = self._mel(clear_wave)
        noicy_spec = self._mel(noicy_wave)
        return refer_spec, clear_spec, noicy_spec