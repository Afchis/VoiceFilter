import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from utils.get_audio import GetAudio


class LibriSpeech300_train(Dataset):
    def __init__(self, epoch_len=1e6):
        super().__init__()
        self.epoch_len = epoch_len
        self.data_path = "./audio/"
        self.get_audio = GetAudio(self.data_path)
        self.trans = transforms.ToTensor()

    def __len__(self):
        return int(self.epoch_len)

    def __getitem__(self, idx):
        refer_spec, clear_spec, noicy_spec = self.get_audio.train_data()
        refer_spec, clear_spec, noicy_spec = self.trans(refer_spec), self.trans(clear_spec), self.trans(noicy_spec)
        return refer_spec, clear_spec, noicy_spec


def Loader(batch_size, num_workers, shuffle=False):
    train_data = LibriSpeech300_train()
    train_loader = DataLoader(dataset=train_data,
                              batch_size=batch_size,
                              num_workers=num_workers,
                              shuffle=False)
    return train_loader
