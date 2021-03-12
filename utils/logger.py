import torch


class Logger():
    def __init__(self, len_train):
        self.epoch = 0
        self.iter = 0
        self.len_train = len_train

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
        elif key == "sim":
            self.disp[key] = x
        else: 
            self.disp[key] += x

    def printer_train(self):
        print(" "*70, end="\r")
        print("Train prosess: [%0.2f" % (100*self.disp["train_iter"]/self.len_train) + chr(37) + "]", "Iter: %s" % self.iter,
              "Loss: %0.2f" % (self.disp["train_loss"]/self.disp["train_iter"]), "sim+-: %0.1f, %0.1f" % (self.disp["sim"][0], self.disp["sim"][1]), end="\r")

    def printer_epoch(self):
        head = "Epoch %s" % self.epoch
        print(" "*70, end="\r")
        print(head, "train:",
              "Loss: %0.2f" % (self.disp["train_loss"]/self.disp["train_iter"]))