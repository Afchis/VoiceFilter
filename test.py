import argparse
import soundfile as sf

import torch

# import class()
from model.model_head import VoiceFilter
from utils.get_audio import GetAudio

# import def()
from dataloader.dataloader import Loader


# init model
model = VoiceFilter()
model.cuda()
model.load_state_dict(torch.load("./weights/checkpoint0.pth"))

train_loader = Loader(batch_size=1, num_workers=0)

data_path = "./audio/"
data_format = ".wav"
get_audio = GetAudio(data_path=data_path, data_format=data_format)


def test():
    model.eval()
    for iter, data in enumerate(train_loader):
        print(iter)
        if iter > 3: break
        refer_spec, clear_spec, noicy_spec = data
        refer_spec, clear_spec, noicy_spec = refer_spec.cuda(), clear_spec.cuda(), noicy_spec.cuda()
        pred_spec = model(refer_spec, noicy_spec)
        noicy_audio = get_audio.inverse_mel(noicy_spec.detach().cpu().numpy()[0, 0])
        clear_audio = get_audio.inverse_mel(clear_spec.detach().cpu().numpy()[0, 0])
        pred_audio = get_audio.inverse_mel(pred_spec.detach().cpu().numpy()[0, 0])
        sf.write(data_path + "noicy_%i" % iter + data_format, noicy_audio, 16000)
        sf.write(data_path + "clear_%i" % iter + data_format, clear_audio, 16000)
        sf.write(data_path + "pred_%i" % iter + data_format, pred_audio, 16000)


if __name__ == "__main__":
    test()