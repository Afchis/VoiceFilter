import torch

# import class()
from model.voice_encoder import SpeechEmbedder
from loss_metric.losses import GE2ELoss

from utils.logger import Logger

# import def()
from dataloader.dvec_dataloader import LibriSpeech300_dvec


# init models:
model = SpeechEmbedder()
model.cuda()

# init dataloader
train_loader = LibriSpeech300_dvec()

# init optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=1)


def train():
    logger = Logger(len_train=len(train_loader))
    for epoch in range(1000):
        logger.init()
        model.train()
        for iter in range(len(train_loader)):
        # for iter, spec in enumerate(train_loader):
            spec = train_loader[iter].cuda()
            loss, pos_sim, neg_sim = model(spec)
            optimizer.zero_grad()
            loss.backward()
            for name, param in model.named_parameters():
                if "proj" in name:
                    param.grad *= 0.5
                elif "loss" in name:
                    param.grad *= 0.01
            torch.nn.utils.clip_grad_norm_(model.lstm.parameters(), 3, norm_type=2)
            optimizer.step()
            logger.update("train_iter", iter)
            logger.update("train_loss", loss.item())
            logger.update("sim", [pos_sim, neg_sim])
            logger.printer_train()
        logger.printer_epoch()


if __name__ == "__main__":
    train()

