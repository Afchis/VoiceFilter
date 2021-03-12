import random
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils.get_audio import GetAudio


class LibriSpeech300_dvec(Dataset):
    def __init__(self, epoch_len=25):
        super().__init__()
        self.epoch_len = epoch_len
        self.data_path = "/workspace/db/audio/Libri/LibriSmall/"
        self.get_audio = GetAudio(self.data_path)
        self.trans = transforms.ToTensor()

    def __len__(self):
        return int(self.epoch_len)

    def __getitem__(self, idx):
        waves = self.get_audio.get_wave_dvec() # waves_list shape --> [self.speaker_num, self.utterance_num]
        spec = list(map(lambda item: self.trans(self.get_audio._wav2mel(item)), waves)) # shape --> [self.speaker_num * self.utterance_num, c, mel, time]
        spec = torch.stack(spec, dim=0)
        return spec.float()

