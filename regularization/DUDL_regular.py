# import libraries
import itertools
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import matplotlib.pyplot as plt
from IPython import display
display.set_matplotlib_formats('svg')

configs = {
    "depth": [1, 3, 5],
    "_params": [None, 128, 512],
    "train_size": [.8],
    "trainBatchSize": [1, 16, 64],
    "dr": [.2, .5, .7, None]
}

# DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DEVICE = 'cpu'


class ClassifierModel(nn.Module):
    def __init__(self, _in, _out, depth=1, _params=None, dr=.5):
        super().__init__()
        self.dr = dr
        self.layers = nn.ModuleList(
            [nn.Linear(_in, _out)] if depth == 1 else [nn.Linear(_in, _params)] +
            [nn.Linear(_params, _params) for _ in range(depth-2)] +
            [nn.Linear(_params, _out)]
        )

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
            x = F.dropout(x, p=self.dr, training=self.training)
        return self.layers[-1](x)


class Classifier():
    def __init__(
            self,
            data, labels,
            lr=.01,  numepochs=500, depth=1, _params=None, train_size=.8,
            trainBatchSize=1,
            testBatchSize=None,
            batchSize=None,
            dr=0):
        if isinstance(batchSize, int):
            trainBatchSize = testBatchSize = batchSize
        _in = 1 if len(data.shape) == 1 else data.shape[1]

        if len(labels.shape) != 1:
            out = labels.shape[1]
        elif labels.max() != 1:
            out = labels.max()+1
        else:
            out = 1
        # model architecture
        self.lr = lr
        self.numepochs = numepochs
        self.model = ClassifierModel(
            _in, out, depth, _params, dr).to(device=DEVICE)
        # loss and labeling function
        if out == 1:
            self.lossfun = nn.BCEWithLogitsLoss()
            self.labelfun = lambda preds: preds > 0
            # self.lossfun = nn.BCELoss()
            # self.labelfun = lambda preds: preds > .5

        else:
            self.lossfun = nn.CrossEntropyLoss()
            self.labelfun = lambda preds: torch.argmax(preds, axis=1)
        # optimizer
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.lr)

        trainData, testData, trainLabels, testLabels = train_test_split(
            data, labels, train_size=train_size)
        if testBatchSize == None:
            testBatchSize = len(testLabels)
        trainDataset = TensorDataset(trainData, trainLabels)
        testDataset = TensorDataset(testData, testLabels)

        self.numOfTrainBatches = np.math.ceil(
            len(trainLabels) / trainBatchSize)
        self.numOfTestBatches = np.math.ceil(len(testLabels) / testBatchSize)
        self.trainLoader = DataLoader(trainDataset, batch_size=trainBatchSize)
        self.testLoader = DataLoader(testDataset, batch_size=testBatchSize)

    def loop(self):
        self.trainaccs = torch.zeros(self.numepochs)
        self.testaccs = torch.zeros(self.numepochs)
        self.trainlosses = torch.zeros(self.numepochs)
        self.testlosses = torch.zeros(self.numepochs)
        for epochi in range(self.numepochs):
            self.trainaccs[epochi], self.trainlosses[epochi] = self.train()
            self.testaccs[epochi], self.testlosses[epochi] = self.test()
        return self.trainaccs, self.testaccs, self.trainlosses, self.testlosses

    def train(self):
        batchaccacc = 0
        batchlossacc = 0
        # enable regularization
        self.model.train()
        for X, y in self.trainLoader:
            X, y = X.to(device=DEVICE, non_blocking=True), y.to(
                device=DEVICE, non_blocking=True)
            # forward pass
            yHat = self.model(X)

            # compute loss
            loss = self.lossfun(yHat, y)
            batchlossacc += loss
            batchaccacc += self.getAcc(yHat, y)

            # backprop
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
        return batchaccacc / self.numOfTrainBatches, batchlossacc / self.numOfTrainBatches

    def test(self):
        batchaccacc = 0
        batchlossacc = 0
        # disable regularization
        self.model.eval()
        with torch.inference_mode():
            for X, y in self.testLoader:
                X, y = X.to(device=DEVICE, non_blocking=True), y.to(
                    device=DEVICE, non_blocking=True)
                # forward pass
                yHat = self.model(X)

                # compute loss
                batchlossacc += self.lossfun(yHat, y)
                batchaccacc += self.getAcc(yHat, y)

        return batchaccacc / self.numOfTestBatches, batchlossacc / self.numOfTestBatches

    def split(self, data, labels, partitions):
        """ partitions: order is train,devset,test. It can be either a list of 2 or 3 elements """
        # split the data (note the third input, and the TMP in the variable name)
        train_data, testTMP_data, train_labels, testTMP_labels = train_test_split(
            data, labels, train_size=partitions[0])

        # now split the TMP data
        split = partitions[1] / (1-partitions[0])
        devset_data, test_data, devset_labels, test_labels = train_test_split(
            testTMP_data, testTMP_labels, train_size=split)
        return train_data, devset_data, test_data, train_labels, devset_labels, test_labels

    def getAcc(self, preds, labels):
        return 100*(self.labelfun(preds) == labels).float().mean()

    def print(self):
        print(self.model)
        for name, param in self.model.named_parameters():
            print(name, param.shape, param.numel())

    def load(self, path):
        self.model.load_state_dict(torch.load(path))

    def save(self, path):
        torch.save(self.model.state_dict(), path)


def factorsList(N):
    arr = torch.arange(1, N+1)
    return arr[N % arr == 0]


def experiment(data, labels, times=1, **kwargs):
    numepochs = kwargs["numepochs"] if "numepochs" in kwargs.keys() else 100
    kwargs["numepochs"] = numepochs
    trainaccs = torch.zeros(numepochs)
    testaccs = torch.zeros(numepochs)
    trainlosses = torch.zeros(numepochs)
    testlosses = torch.zeros(numepochs)

    for _ in range(times):
        classifier = Classifier(data, labels, **kwargs)
        trainaccstmp, testaccstmp, trainlossestmp, testlossestmp = classifier.loop()
        trainaccs += trainaccstmp
        testaccs += testaccstmp
        trainlosses += trainlossestmp
        testlosses += testlossestmp

    return trainaccs/times, testaccs/times, trainlosses/times, testlosses/times


def validArgs(data, labels, **kwargs):
    return (
        xor(kwargs["depth"] == 1, kwargs["_params"] != None)
        if "depth" in kwargs and "_params" in kwargs else True and

        (kwargs["trainBatchSize"] <= kwargs["train_size"] * len(labels))
        if "train_size" in kwargs else True
    )


def xor(p, q): return (p and not q) or (not p and q)


# https://stackoverflow.com/a/5228294/10713877


def product_dict(condition=lambda _: True, data=None, labels=None, **kwargs):
    keys = kwargs.keys()
    vals = kwargs.values()
    for instance in itertools.product(*vals):
        config = dict(zip(keys, instance))
        if condition(data, labels, **config):
            yield config


def printConfigs(**kwargs):
    for i, kwarg in enumerate(product_dict(condition=validArgs,  **kwargs)):
        print(i, kwarg)


def iterConfigsWithNumepochs(numepochs=100, times=1, data=None, labels=None,  **configs):
    listOfConfigs = list(product_dict(condition=validArgs,
                         data=data, labels=labels, **configs))
    lenConfigs = len(listOfConfigs)

    trainaccs = torch.zeros(lenConfigs, numepochs)
    testaccs = torch.zeros(lenConfigs, numepochs)
    trainlosses = torch.zeros(lenConfigs, numepochs)
    testlosses = torch.zeros(lenConfigs, numepochs)

    for i, kwargs in enumerate(listOfConfigs):
        trainaccsacc = torch.zeros(numepochs)
        testaccsacc = torch.zeros(numepochs)
        trainlossesacc = torch.zeros(numepochs)
        testlossesacc = torch.zeros(numepochs)
        for _ in range(times):
            classifier = Classifier(
                data, labels, numepochs=numepochs, **kwargs)
            trainaccstmp, testaccstmp, trainlossestmp, testlossestmp = classifier.loop()
            trainaccsacc += trainaccstmp
            testaccsacc += testaccstmp
            trainlossesacc += trainlossestmp
            testlossesacc += testlossestmp
        trainaccs[i] = trainaccsacc / times
        testaccs[i] = testaccsacc / times
        trainlosses[i] = trainlossesacc / times
        testlosses[i] = testlossesacc / times

    return trainaccs, testaccs, trainlosses, testlosses, listOfConfigs


def plot2d(trainvals, testvals, v, trainTitle="Training accuracy", testTitle="Test accuracy"):
    numepochs = trainvals.shape[1]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    ax[0].imshow(trainvals, aspect='auto', origin="lower",
                 vmin=50, vmax=90, extent=[0, numepochs, v[0], v[-1]])
    ax[0].set_xlabel('Epochs')
    ax[0].set_ylabel('Parameter index')
    ax[0].set_title(trainTitle)

    p = ax[1].imshow(testvals, aspect='auto', origin="lower",
                     vmin=50, vmax=90, extent=[0, numepochs, v[0], v[-1]])
    ax[1].set_xlabel('Epochs')
    ax[1].set_ylabel('Parameter index')
    ax[1].set_title(testTitle)
    fig.colorbar(p, ax=ax[1])

    plt.show()


def plot2din1d(trainvals, testvals, trainTitle="Training accuracy", testTitle="Test accuracy"):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))

    for trainval in trainvals:
        ax[0].plot(trainval)
    ax[0].legend(np.arange(len(trainvals)))
    ax[0].set_xlabel('Epochs')
    ax[0].set_title(trainTitle)

    for testval in testvals:
        ax[1].plot(testval)
    ax[1].legend(np.arange(len(testvals)))
    ax[1].set_xlabel('Epochs')
    ax[1].set_title(testTitle)

    plt.show()


def plot(trainvals, testvals, title="Accuracy"):
    newtrainvals = trainvals if isinstance(
        trainvals, np.ndarray) else trainvals.detach().numpy()
    newtestvals = testvals if isinstance(
        testvals, np.ndarray) else testvals.detach().numpy()
    plt.plot(newtrainvals)
    plt.plot(newtestvals)
    plt.xlabel('Epochs')
    plt.ylabel(title)
    plt.legend(["train", "test"])
    plt.title(f"Train = {trainvals[-1]}, Test = {testvals[-1]}")
    plt.show()
# create a 1D smoothing filter


def getmax(accs):
    amax = int(accs.argmax())
    x = amax % accs.shape[1]
    y = amax // accs.shape[1]
    return y, x


def meanFilter(x, k=5, mode="valid"):
    return x.detach().numpy() if k == 0 else np.convolve(x.detach().numpy(), np.ones(k)/k, mode=mode)


def meanFilter2d(x, **kwargs):
    return np.array([meanFilter(_x, **kwargs) for _x in x])
