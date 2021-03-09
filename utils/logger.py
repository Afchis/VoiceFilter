import torch


class Logger():
    def __init__(self, len_train):
        self.epoch = 0
        self.iter = 0
        self.len_train = len_train
        self.db_level = 0
        self.loss = 0

    def init(self):
        self.epoch += 1
        self.disp = {
            "train_iter" : 0,
            "train_loss" : 0,
        }

    def update(self, key, x):
        if key == "train_iter":
            self.iter += 1
            self.disp[key] += 1
        elif key == "db_level":
            self.db_level = x
        elif key == "loss":
            self.loss = x
        else: 
            self.disp[key] += x

    def printer_train(self):
        print(" "*70, end="\r")
        print("Train prosess: [%0.2f" % (100*self.disp["train_iter"]/self.len_train) + chr(37) + "]", "Iter: %s" % self.iter,
              "Loss: %0.2f" % (self.disp["train_loss"]/self.disp["train_iter"]),
              "min: %0.2f" % self.db_level[0], "max: %0.2f" % self.db_level[1], "%0.2f" % self.loss, end="\r")

    def printer_epoch(self):
        head = "Epoch %s" % self.epoch
        print(" "*70, end="\r")
        print(head, "train:",
              "Loss: %0.2f" % (self.disp["train_loss"]/self.disp["train_iter"]))