import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

import nn
import nn.functional as F


n_features = 28 * 28
n_classes = 10
n_epochs = 10
bs = 1000
lr = 1e-2
lengths = (n_features, bs, n_classes)


class Model(nn.Module):

    # TODO Design the classifier.

    ...
    def __init__(self, lengths):
        self.n = lengths[1]
        self.n_classes = lengths[2]
        self.a0 = np.zeros((1, 1))
        self.conv2d1 = nn.Conv2d(1, 2)
        self.avgpool1 = nn.AvgPool([])
        self.conv2d2 = nn.Conv2d(2, 2)
        self.avgpool2 = nn.AvgPool([])
        self.linear1 = nn.Linear(72, 10)
        self.sigmoid1 = nn.functional.Sigmoid([])
        self.a0 = np.zeros((1,1))
        self.a1 = np.zeros((1,1))
        self.a2 = np.zeros((1,1))
        self.a3 = np.zeros((1,1))
        self.a4 = np.zeros((1,1))
        self.a5 = np.zeros((1,1))
        self.delta1 = np.zeros((1,1))
        self.delta3 = np.zeros((1,1))
        self.delta5 = np.zeros((1,1))

    def forward(self, x: np.ndarray) -> np.ndarray:
        a0 = x = x.reshape((np.shape(x)[0],1,28,28))
        a1 = self.conv2d1.forward(a0)
        a2 = self.avgpool1.forward(a1)
        a3 = self.conv2d2.forward(a2)
        a4 = self.avgpool2.forward(a3)
        a_4 = a4.reshape((np.shape(a4)[0], np.shape(a4)[1]*np.shape(a4)[2]*np.shape(a4)[3]))
        a5 = self.linear1.forward(a_4)
        a6 = self.sigmoid1.forward(a5)
        self.a0 = a0
        self.a1 = a1
        self.a2 = a2
        self.a3 = a3
        self.a4 = a4
        self.a5 = a5
        self.a6 = a6
        self.avgpool1.a = a1
        self.avgpool2.a = a3
        self.sigmoid1.a = a5
        return a6

    def backward(self, dy: np.ndarray) -> np.ndarray:
        delta6 = dy
        delta5 = self.sigmoid1.backward(delta6)
        delta4 = self.linear1.backward(delta5)
        delta_4 = np.reshape(delta4, np.shape(self.a4))
        delta3 = self.avgpool2.backward(delta_4)
        delta2 = self.conv2d2.backward(delta3)
        delta1 = self.avgpool1.backward(delta2)

        self.delta1 = delta1
        self.delta3 = delta3
        self.delta5 = delta5

        return delta1


    # End of todo


def load_mnist(mode='train', n_samples=None, flatten=True):
    images = './data/train-images.idx3-ubyte' if mode == 'train' else './data/t10k-images.idx3-ubyte'
    labels = './data/train-labels.idx1-ubyte' if mode == 'train' else './data/t10k-labels.idx1-ubyte'
    length = 60000 if mode == 'train' else 10000

    X = np.fromfile(open(images), np.uint8)[16:].reshape(
        (length, 28, 28)).astype(np.int32)
    if flatten:
        X = X.reshape(length, -1)
    y = np.fromfile(open(labels), np.uint8)[8:].reshape(
        (length)).astype(np.int32)
    return (X[:n_samples] if n_samples is not None else X,
            y[:n_samples] if n_samples is not None else y)


def vis_demo(model):
    X, y = load_mnist('test', 20)
    probs = model.forward(X)
    preds = np.argmax(probs, axis=1)
    fig = plt.subplots(nrows=4, ncols=5, sharex='all',
                       sharey='all')[1].flatten()
    for i in range(20):
        img = X[i].reshape(28, 28)
        fig[i].set_title(preds[i])
        fig[i].imshow(img, cmap='Greys', interpolation='nearest')
    fig[0].set_xticks([])
    fig[0].set_yticks([])
    plt.tight_layout()
    plt.savefig("vis.png")
    plt.show()


def rot(w):
    new_arr = np.reshape(w, len(w) * len(w[0]))
    new_arr = new_arr[::-1]
    new_arr = np.reshape(new_arr, np.shape(w))
    return new_arr

def convolution(delta:np.array, a:np.array):
    delta = np.reshape(delta, (1, np.shape(delta)[0], np.shape(delta)[1]))
    c_in1, h_in1, w_in1 = np.shape(delta)[0], np.shape(delta)[1], np.shape(delta)[2]
    c_in2, h_in2, w_in2 = np.shape(a)[0], np.shape(a)[1], np.shape(a)[2]
    res = np.dot(Conv_im2col(delta, a), Conv_im2col(delta, delta).T)
    return res.reshape((c_in2-c_in1+1, h_in2-h_in1+1, w_in2-w_in1+1))

def Conv_im2col(delta:np.array, a:np.array):
    c_in1, h_in1, w_in1 = len(delta), len(delta[0]), len(delta[0][0])
    c_in2, h_in2, w_in2 = len(a), len(a[0]), len(a[0][0])
    image_col = []
    for k in range(0, c_in2 - c_in1 + 1):
        for i in range(0, h_in2 - h_in1 + 1):
            for j in range(0, w_in2 - w_in1 + 1):
                col = a[k:k + c_in1, i:i + h_in1, j:j + w_in1].reshape([-1])
                image_col.append(col)
    return np.array(image_col)

def main():
    trainloader = nn.data.DataLoader(load_mnist('train'), batch=bs)
    testloader = nn.data.DataLoader(load_mnist('test',10))
    model = Model(lengths)
    optimizer = nn.optim.SGD(model, lr=lr, momentum=0.9)
    criterion = F.CrossEntropyLoss(n_classes=n_classes)
    # for X,y in trainloader:
    #     probs1 = model.forward(X)
    #     model.linear1.w[1][1]+=0.01
    #     probs2 = model.forward(X)
    #     loss1 = criterion(probs1, y)
    #     loss2 = criterion(probs2, y)
    #     model.backward(criterion.backward())
    #     dw3 = np.zeros((73, 10))
    #     a4 = model.a4
    #     a4 = a4.reshape((np.shape(a4)[0], np.shape(a4)[1] * np.shape(a4)[2] * np.shape(a4)[3]))
    #     dw3[1:] = np.dot(a4.T, model.delta5) * model.n
    #     dw3[0] = np.sum(model.delta5, axis=0)
    #     print(dw3[1][1])
    #     print((loss2-loss1)/0.01)


    for i in range(n_epochs):
        bar = tqdm(trainloader, total=6e4/bs)
        bar.set_description(f'epoch  {i:2}')
        for X, y in bar:
            probs = model.forward(X)
            loss = criterion(probs, y)
            model.backward(criterion.backward())
            optimizer.step()
            preds = np.argmax(probs, axis=1)
            bar.set_postfix_str(f'acc={np.sum(preds == y) / len(y) * 100:.1f}'
                                f' loss={loss:.3f}')

        for X, y in testloader:
            probs = model.forward(X)
            preds = np.argmax(probs, axis=1)
            print(f' test acc: {np.sum(preds == y) / len(y) * 100:.1f}')

    vis_demo(model)


if __name__ == '__main__':
    main()
