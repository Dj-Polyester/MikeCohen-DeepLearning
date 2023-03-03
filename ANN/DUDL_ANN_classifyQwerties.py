import torch
import torch.nn as nn
import numpy as np
# import multiprocessing as mp
# import os
PATH = "qwerty_multilayer.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

NPERCLUST = 100
# create data
BLUR = 2

A = [1, 1]
B = [5, 1]


def createData():
    # generate data
    # a = [A[0]+np.random.randn(NPERCLUST)*BLUR, A[1] +
    #      np.random.randn(NPERCLUST)*BLUR]
    # b = [B[0]+np.random.randn(NPERCLUST)*BLUR, B[1] +
    #      np.random.randn(NPERCLUST)*BLUR]
    # # concatanate into a matrix
    # data_np = np.hstack((a, b)).T
    # data = torch.tensor(data_np).float()

    # true labels
    labels_np = np.vstack((np.zeros((NPERCLUST, 1)), np.ones((NPERCLUST, 1))))

    # convert to a pytorch tensor
    labels = torch.tensor(labels_np).float()
    return labels


class QwertyClassifier():
    def __init__(self, learningRate, numepochs):
        # build the model

        # self.model = nn.Sequential(
        #     nn.Linear(2, 1),   # input layer
        #     nn.ReLU(),        # activation unit
        #     nn.Linear(1, 1),   # output unit
        #     # nn.Sigmoid(),     # final activation unit (here for conceptual reasons; in practice, better to use BCEWithLogitsLoss)
        # )  # .to(DEVICE)

        self.model = nn.Sequential(
            nn.Linear(2, 16),  # input layer
            nn.ReLU(),        # activation unit
            nn.Linear(16, 1),  # hidden layer
            nn.ReLU(),        # activation unit
            nn.Linear(1, 1),   # output unit
            # nn.Sigmoid(),     # final activation unit
        )  # .to(DEVICE)

        # self.init_weights()

        self.lossfun = nn.BCEWithLogitsLoss()
        self.numepochs = numepochs
        self.learningRate = learningRate
        # Note: You'll learn in the "Metaparameters" section that it's better to use BCEWithLogitsLoss, but this is OK for now.
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

            # predlabels = predictions>.5
            predlabels = predictions > 0

            # find errors
            misclassified = torch.where(predlabels != labels)[0]

            # total accuracy
            totalacc = 100-100*len(misclassified)/(2*NPERCLUST)
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


labels = createData()

data = torch.tensor([[3.6012e-01,  7.6464e-01],
                     [5.5072e+00,  4.2577e-01],
                     [8.0227e-01,  2.1193e+00],
                     [2.4405e+00, -3.4805e+00],
                     [7.4319e-01,  1.2416e+00],
                     [2.5452e+00, -3.9293e-01],
                     [1.6299e+00,  2.8289e+00],
                     [2.7857e+00,  2.2383e+00],
                     [2.2661e-01,  5.1229e-01],
                     [2.3635e-01,  2.7733e+00],
                     [5.1923e-01, -1.0651e+00],
                     [3.5399e+00,  1.1905e+00],
                     [-3.3637e-01,  1.6072e+00],
                     [5.7524e-01,  2.3180e+00],
                     [-3.1539e-01,  1.5188e+00],
                     [2.3467e-02, -8.3744e-01],
                     [3.0668e+00,  2.9577e+00],
                     [1.3314e+00, -8.1402e+00],
                     [1.9726e+00,  3.4702e+00],
                     [1.8306e+00,  8.4110e-01],
                     [1.9390e+00,  1.0383e+00],
                     [4.1460e+00,  9.3302e-01],
                     [8.9091e-01,  2.4174e+00],
                     [-5.9211e+00,  9.8307e-01],
                     [-1.0122e+00,  3.0036e-01],
                     [1.8931e+00, -2.5844e+00],
                     [2.3175e+00,  1.4166e+00],
                     [2.0184e+00, -8.8236e-02],
                     [4.7546e+00,  3.5685e+00],
                     [3.4930e+00,  3.5859e+00],
                     [2.5833e+00, -7.7174e-01],
                     [2.8107e+00, -2.0279e+00],
                     [2.5927e+00,  7.2255e-01],
                     [1.9113e+00,  1.1359e+00],
                     [4.3130e+00, -5.5202e-01],
                     [1.7509e-01,  4.8103e-01],
                     [-9.4677e-01,  2.1784e+00],
                     [3.5433e+00,  1.0590e+00],
                     [4.9357e+00,  2.0517e-01],
                     [-1.1399e+00,  1.1017e+00],
                     [-3.8865e+00,  1.2471e-01],
                     [1.4592e+00,  1.3925e+00],
                     [-2.7396e+00,  1.2933e+00],
                     [5.3232e-01,  2.9714e+00],
                     [1.6063e+00, -3.7931e-01],
                     [2.4747e+00,  2.1016e+00],
                     [2.2927e+00, -3.2792e+00],
                     [3.2008e+00,  1.5402e+00],
                     [2.2638e+00,  2.9106e+00],
                     [-1.1816e+00,  2.9508e+00],
                     [-1.6694e+00,  2.6283e+00],
                     [3.2355e+00,  2.0483e+00],
                     [2.0271e+00, -3.4167e+00],
                     [2.6473e-01,  1.9151e+00],
                     [-7.0154e-01, -1.6771e+00],
                     [6.1957e+00,  1.4834e+00],
                     [7.3530e-01,  1.2817e+00],
                     [1.6511e+00,  2.5631e+00],
                     [2.6544e+00, -8.3870e-01],
                     [-6.3161e-01,  2.0869e+00],
                     [1.2483e+00, -1.3990e+00],
                     [-1.1369e+00,  3.7811e+00],
                     [-1.9016e+00,  2.9709e+00],
                     [5.5965e-01,  6.6481e-01],
                     [1.6081e+00,  3.4199e+00],
                     [3.4920e+00, -1.8362e+00],
                     [7.5666e-01,  4.9260e-01],
                     [-2.0518e-01, -3.1810e+00],
                     [1.2727e+00,  1.3367e+00],
                     [3.3626e+00,  3.6037e-01],
                     [1.9767e+00,  1.4359e+00],
                     [2.5277e+00,  3.1372e-01],
                     [-1.2996e+00, -1.0787e+00],
                     [1.6486e+00,  2.8228e+00],
                     [1.2659e+00,  2.8785e+00],
                     [6.4264e+00,  9.0293e-01],
                     [2.2606e+00, -8.8184e-02],
                     [1.9733e+00,  2.6261e+00],
                     [-3.1121e+00,  1.2098e+00],
                     [1.4338e-01,  2.8732e+00],
                     [-2.2492e+00,  3.3800e-01],
                     [-4.5722e+00, -1.1709e+00],
                     [-1.0402e+00, -2.6707e+00],
                     [-1.6128e-01,  2.4068e+00],
                     [-7.7478e-01,  4.3540e+00],
                     [1.4085e+00,  9.4303e-01],
                     [1.4493e+00, -2.6099e+00],
                     [-1.2519e+00,  1.8125e+00],
                     [9.4343e-01, -9.9506e-01],
                     [-9.4861e-01,  1.9035e+00],
                     [3.3397e+00,  2.7379e+00],
                     [4.8828e-01, -1.1595e+00],
                     [4.1568e-01,  1.7967e+00],
                     [1.9216e+00,  1.0408e+00],
                     [7.8129e-02,  2.4362e-01],
                     [-9.6619e-01,  1.6553e-01],
                     [2.2388e+00, -9.8534e-01],
                     [-2.3961e-01, -5.6279e-01],
                     [-9.0973e-02, -1.6387e-01],
                     [-6.1899e-02,  2.1404e+00],
                     [7.2469e+00,  6.2747e+00],
                     [5.2452e+00,  3.7537e+00],
                     [3.7503e+00, -1.9767e+00],
                     [4.7382e+00,  6.5475e-01],
                     [3.8566e+00,  2.1007e+00],
                     [8.0787e+00, -1.9706e+00],
                     [5.7723e+00,  1.3478e+00],
                     [4.8131e+00, -1.2710e+00],
                     [4.4810e+00,  1.9551e+00],
                     [4.9855e+00,  8.4840e-02],
                     [5.1796e+00,  1.4675e+00],
                     [5.1149e+00,  2.2759e+00],
                     [5.9715e+00,  8.2414e-01],
                     [5.2311e+00, -7.7853e-01],
                     [7.2342e+00, -3.5422e+00],
                     [4.9598e+00,  8.2204e+00],
                     [7.1612e+00,  2.4185e+00],
                     [1.9729e+00,  1.4890e+00],
                     [6.0530e+00,  2.1340e+00],
                     [2.1624e+00, -2.8966e-01],
                     [7.2343e+00, -9.4330e-01],
                     [4.3432e+00,  1.7255e+00],
                     [4.1339e+00, -2.7287e-01],
                     [4.4702e+00,  6.9231e-02],
                     [6.0876e+00, -2.8401e-01],
                     [1.1684e+00, -3.5661e+00],
                     [5.9617e+00,  6.7776e-01],
                     [6.9178e+00, -4.1740e-01],
                     [1.9859e+00,  1.2790e+00],
                     [2.8203e+00,  7.1437e-01],
                     [3.5414e+00,  4.9518e-01],
                     [5.3533e+00, -6.6293e-01],
                     [1.0987e+00,  2.7858e+00],
                     [3.1657e+00,  2.8413e+00],
                     [3.6703e+00,  1.8895e+00],
                     [5.2291e+00,  4.4104e+00],
                     [2.7475e+00,  2.1060e+00],
                     [8.6918e+00, -6.7504e-01],
                     [5.2561e+00,  2.3083e+00],
                     [3.8656e+00,  1.7551e+00],
                     [5.3998e+00, -6.3486e-01],
                     [5.4947e+00, -9.6498e-04],
                     [3.0484e+00,  2.9047e+00],
                     [4.4420e+00,  1.4503e+00],
                     [3.6553e+00,  1.5932e+00],
                     [5.3332e+00,  2.1328e+00],
                     [5.4927e+00,  4.0500e+00],
                     [6.0712e+00, -2.4568e+00],
                     [5.9004e+00,  1.5814e+00],
                     [4.8380e+00,  2.0595e+00],
                     [3.4676e+00,  2.8450e+00],
                     [2.1632e+00,  1.2659e+00],
                     [3.8667e+00,  5.0456e+00],
                     [5.4229e+00, -1.5745e+00],
                     [6.1993e+00, -1.1239e+00],
                     [8.3373e+00,  5.3713e-01],
                     [2.0684e+00,  2.3571e+00],
                     [3.1398e+00,  4.1396e-01],
                     [5.4957e+00,  1.9575e+00],
                     [8.0174e+00,  2.8320e+00],
                     [9.0872e+00,  4.4451e+00],
                     [7.3496e+00,  3.2344e+00],
                     [7.5824e+00, -1.7419e+00],
                     [7.1107e+00, -1.4490e+00],
                     [2.7549e+00,  7.5576e-01],
                     [7.7014e+00,  1.5320e-01],
                     [6.2819e+00,  9.2767e-01],
                     [4.0477e+00, -3.5289e-01],
                     [5.3679e+00,  3.4568e+00],
                     [4.1061e+00,  8.6618e-02],
                     [3.3321e+00, -1.2768e+00],
                     [5.0002e+00,  1.1689e+00],
                     [7.4581e+00,  6.2752e-01],
                     [5.9759e+00,  6.4744e-02],
                     [4.4028e+00, -7.7171e-01],
                     [6.1981e+00,  3.2629e+00],
                     [6.1594e+00,  2.4269e+00],
                     [2.9356e+00,  1.6152e+00],
                     [7.7744e+00,  4.2183e+00],
                     [5.1936e+00,  1.8682e+00],
                     [5.5423e+00, -6.6557e-01],
                     [9.4319e+00, -4.0566e+00],
                     [5.6549e+00,  4.5452e+00],
                     [6.2043e+00, -2.8550e-01],
                     [3.9954e+00,  2.0055e+00],
                     [6.4413e+00,  1.4010e+00],
                     [3.9999e+00,  4.3146e-01],
                     [5.0797e-01,  5.7600e-01],
                     [6.5650e+00,  1.6507e+00],
                     [4.4231e+00,  1.3397e+00],
                     [4.8120e+00,  3.0701e+00],
                     [3.5438e+00, -2.1159e+00],
                     [5.0811e+00, -5.3436e-01],
                     [5.2156e+00,  3.6905e-03],
                     [4.8668e+00,  2.3694e+00],
                     [3.8346e+00,  9.9745e-01],
                     [6.8607e+00,  3.3768e+00],
                     [9.2040e+00,  2.8467e-01],
                     [5.3268e+00,  1.9659e+00],
                     [7.8206e+00,  4.0111e+00]])
