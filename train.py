import argparse

import torch

# import class()
from model.model_head import VoiceFilter

# import def()
from dataloader.dataloader import Loader


# init model
model = VoiceFilter()

# init dataloader
train_loader = Loader(batch_size=1, num_workers=0)

# init optimizer
optimizer = torch.optim.SGD(model.parameters(), lr=0.005)

# loss
criterion = torch.nn.MSELoss()


def train():
    for epoch in range(10):
        model.train()
        for iter, data in enumerate(train_loader):
            refer_spec, clear_spec, noicy_spec = data
            pred_spec = model(refer_spec, noicy_spec.float())
            loss = criterion(pred_spec, clear_spec.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(loss.item())




if __name__ == "__main__":
    train()