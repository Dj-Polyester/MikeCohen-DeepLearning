import torch
import torch.nn as nn
import numpy as np
# import multiprocessing as mp
# import os
PATH = "qwerty_multioutput.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NPERCLUST = 100
# create data
BLURS = [
    .5,
    1,
    .7,
]
A = torch.tensor(
    [
        [1, 1],
        [5, 1],
        [3, 3],
    ]
)
N = len(A)


def createData():
    az0 = torch.concat(
        [a+torch.randn(NPERCLUST)*BLURS[i] for i, a in enumerate(A[:, 0])])
    az1 = torch.concat(
        [a+torch.randn(NPERCLUST)*BLURS[i] for i, a in enumerate(A[:, 1])])
    data = torch.stack((az0, az1), axis=1)

    # true labels
    labels = torch.concat([torch.full((NPERCLUST,), i) for i in range(N)])

    return data, labels


class QwertyClassifier():
    def __init__(self, learningRate, numepochs):
        # build the model

        self.model = nn.Sequential(
            nn.Linear(2, 4),  # input layer
            nn.ReLU(),        # activation unit
            nn.Linear(4, 3),  # hidden layer
            nn.ReLU(),        # activation unit
            nn.Linear(3, 3),   # output unit
        )  # .to(DEVICE)

        # self.init_weights()

        self.lossfun = nn.CrossEntropyLoss()
        self.numepochs = numepochs
        self.learningRate = learningRate
        self.optimizer = torch.optim.SGD(
            self.model.parameters(), lr=self.learningRate)

    def init_weights(self):
        if isinstance(self.model, nn.Linear):
            nn.init.kaiming_uniform(self.model.weight, nonlinearity='relu')
            self.model.bias.data.fill_(0.01)

    def train(self, data, labels):
        self.model.train()
        self.losses = torch.zeros(self.numepochs)

        for epochi in range(self.numepochs):
            # forward pass
            yHat = self.model(data)

            # compute loss
            loss = self.lossfun(yHat, labels)
            self.losses[epochi] = loss
            # zero-grad
            self.optimizer.zero_grad()
            # backprop
            loss.backward()
            # gradient-descent step
            self.optimizer.step()

    def eval(self, data, labels):
        self.model.eval()
        with torch.inference_mode():
            predictions = self.model(data)

            predlabels = torch.argmax(predictions, axis=1)

            # find errors
            misclassified = torch.where(predlabels != labels)[0]

            # total accuracy
            totalacc = 100-100*len(misclassified)/(N*NPERCLUST)
            return predlabels, misclassified, totalacc


def test(lr, numofepochs, times=1):
    numofepochs = int(numofepochs)
    losses = torch.zeros(times, numofepochs)
    accs = torch.zeros(times)

    for i in range(times):
        classifier = QwertyClassifier(
            lr, numofepochs)
        classifier.train(data, labels)
        losses[i] = classifier.losses
        predlabels, misclassified, totalacc = classifier.eval(
            data, labels)
        accs[i] = totalacc
    return accs.nanmean(), losses.nanmean(axis=0)


TIMES = 50


def calcAcc(lr, epoch):
    totalacc, losses = test(lr, epoch, TIMES)
    return totalacc


def calcAccs(lrs, epochs):
    NUMEPOCHS = epochs.shape[0]
    NUMOFLRS = lrs.shape[0]
    # global data, labels
    # data = data.to(DEVICE)
    # labels = labels.to(DEVICE)
    # if multithread:
    #     y, x = torch.meshgrid(epochs, lrs)
    #     lrepoch = torch.stack((x, y), axis=2).reshape(NUMEPOCHS*NUMOFLRS, 2)
    #     del x, y

    #     mp.set_start_method('spawn')
    #     with mp.Pool(os.cpu_count()) as p:
    #         accs = torch.tensor(p.starmap(calcAcc, lrepoch)
    #                             ).reshape(NUMEPOCHS, NUMOFLRS)
    # else:
    accs = torch.zeros(NUMEPOCHS, NUMOFLRS)
    for i, epoch in enumerate(epochs):
        for j, lr in enumerate(lrs):
            accs[i, j] = calcAcc(lr, epoch)
    return accs


data, labels = createData()
