import argparse
import soundfile as sf
import time

import torch

# import class()
from model.model_head import VoiceFilter
from utils.logger import Logger
from utils.get_audio import GetAudio

# import def()
from dataloader.dataloader import Loader


data_path = "./audio/"
data_format = ".wav"
get_audio = GetAudio(data_path=data_path, data_format=data_format)


# init model
model = VoiceFilter()
model.cuda()

def save_model(epoch):
    torch.save(model.state_dict(), "weights/checkpoint%i.pth" % epoch)


# init dataloader
train_loader = Loader(batch_size=16, num_workers=8)


# init optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)


# loss
criterion = torch.nn.MSELoss(reduction='sum')


def train():
    logger = Logger(len_train=len(train_loader))
    for epoch in range(1000):
        logger.init()
        model.train()
        for iter, data in enumerate(train_loader):
            refer_spec, clear_spec, noicy_spec = data
            refer_spec = list(map(lambda item: item.cuda(), refer_spec))
            clear_spec, noicy_spec = clear_spec.cuda(), noicy_spec.cuda()
            pred_spec = model(refer_spec, noicy_spec)
            loss = criterion(pred_spec, clear_spec)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            logger.update("train_iter", iter)
            logger.update("train_loss", loss.item())
            logger.update("loss", loss.item())
            logger.update("db_level", [pred_spec.min().item(), pred_spec.max().item()])
            logger.printer_train()
        logger.printer_epoch()
        save_model(epoch)
        noicy_audio = get_audio.spec2wav(noicy_spec.detach().cpu().numpy()[0, 0])
        clear_audio = get_audio.spec2wav(clear_spec.detach().cpu().numpy()[0, 0])
        pred_audio = get_audio.spec2wav(pred_spec.detach().cpu().numpy()[0, 0])

        sf.write(data_path + ("clear_%i" % epoch) + data_format, clear_audio, 16000)
        time.sleep(1)

        sf.write(data_path + ("pred_%i" % epoch) + data_format, pred_audio, 16000)
        time.sleep(1)
        
        sf.write(data_path + ("noicy_%i" % epoch) + data_format, noicy_audio, 16000)
        time.sleep(1)
        
        
        


if __name__ == "__main__":
    train()

