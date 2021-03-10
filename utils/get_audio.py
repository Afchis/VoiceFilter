import os, glob, random

import librosa
import numpy as np


class GetAudio():
    def __init__(self, data_path, data_format="*/*.flac"):
        self.data_path = data_path
        self.data_format = data_format
        self.sampling_rate = 16000
        self.wave_len = 48000
        self.n_fft = 1000
        self.num_mels = 128
        self.mels_list = [100, 120, 130, 150]
        self.mel_basis = librosa.filters.mel(sr=self.sampling_rate,
                                             n_fft=512,
                                             n_mels=40)

    def _get_wave(self, id_names):
        id_name = random.sample(id_names, 2)
        target_files = random.sample(glob.glob(os.path.join(self.data_path, id_name[0], self.data_format)), 2)
        inter_file = random.choice(glob.glob(os.path.join(self.data_path, id_name[1], self.data_format)))
        refer_wave, _ = librosa.load(target_files[0][:-5] + "-norm.wav", sr=self.sampling_rate)
        clear_wave, _ = librosa.load(target_files[1], sr=self.sampling_rate)
        inter_wave, _ = librosa.load(inter_file, sr=self.sampling_rate)
        return refer_wave, clear_wave, inter_wave

    def _mix_wave(self, x1, x2):
        x1_time = random.randrange(len(x1) - self.wave_len)
        x2_time = random.randrange(len(x2) - self.wave_len)
        x1 = x1[x1_time:x1_time+self.wave_len]
        x2 = x1 + x2[x2_time:x2_time+self.wave_len]
        return x1, x2

    def _mel_encoder(self, x):
        spec = librosa.core.stft(x, n_fft=512)
        magnitudes = np.abs(spec) ** 2
        mel = np.log10(np.dot(self.mel_basis, magnitudes) + 1e-6)
        return mel

    def _norm(self, x):
        return np.clip((x / 100.), a_min=-1., a_max=0.) + 1.

    def _denorm(self, x):
        return (x - 1.) * 100.
        
    def _wav2spec(self, x):
        out = librosa.stft(x, n_fft=self.n_fft)
        out = librosa.amplitude_to_db(abs(out), ref=1., amin=1e-05, top_db=None) - 20.
        out = self._norm(out)
        return out

    def spec2wav(self, x):
        out = self._denorm(x)
        out = librosa.db_to_amplitude((out + 20.), ref=1.)
        out = librosa.griffinlim(out)
        return out

    # def _mel(self, x):
    #     out = librosa.feature.melspectrogram(x, sr=self.sampling_rate,  n_mels=self.num_mels)
    #     out = librosa.power_to_db(out, ref=np.max)
    #     return out

    # def _mel_stack(self, x):
    #     mels = list()
    #     for i in range(self.mels_list):
    #         mels.append(librosa.feature.melspectrogram(x, sr=self.sampling_rate,  n_mels=self.mels_list[i]))
    #     out = np.concatenate(mels)
    #     return out

    def train_data(self):
        _, id_names, _ = next(os.walk(self.data_path))
        clear_len, inter_len = 0, 0
        while clear_len <= (self.wave_len) or inter_len <= (self.wave_len ):
            refer_wave, clear_wave, inter_wave = self._get_wave(id_names)
            clear_len = len(clear_wave)
            inter_len = len(inter_wave)
        clear_wave, noicy_wave = self._mix_wave(clear_wave, inter_wave)
        # refer_spec = self._stft(refer_wave)
        refer_spec = self._mel_encoder(refer_wave)
        clear_spec = self._wav2spec(clear_wave)
        noicy_spec = self._wav2spec(noicy_wave)
        return refer_spec, clear_spec, noicy_spec