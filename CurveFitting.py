# coding:utf-8

import numpy as np
from PIL import Image
import os
from math import *
from numpy import *
import matplotlib.pyplot as plot


# 每层的神经元个数向量M，用元组记录，因为不可动态改变
M = (1, 8, 1)

# 网络层数len(M)
L = len(M)

# learningRate
rate = 0.01


# 激活函数选择
def activate(x):
    return sigmoid(x)


def dActivate(x):
    return dSigmoid(x)


# 损失函数选择
def lossFunc(output, label):
    return 0


# 权重W
W = []
# 偏置b
b = []

# 用来存放一次迭代中，每一个样本图像得出的梯度值
dw = np.empty(L - 1, dtype=list)
db = np.empty(L - 1, dtype=list)


# sigmoid函数
def sigmoid(x):
    if type(x) != int:
        y = np.zeros(shape(x))
        for i in range(len(x)):
            y[i] = 1 / (1 + exp((-1) * x[i]))
    else:
        y = 1 / (1 + exp(-x))
    return y


# sigmoid函数的导数
def dSigmoid(x):
    if type(x) != int:
        d = np.zeros(shape(x))
        for i in range(len(x)):
            d[i] = sigmoid(x[i]) * (1 - sigmoid(x[i]))
    else:
        d = sigmoid(x) * (1 - sigmoid(x))
    return d


def wbInit(M):
    for i in range(L - 1):
        # weightMatrix = mat(np.full((M[i], M[i+1]), 1))
        weightMatrix = mat(random.uniform(-1 / M[i], 1 / M[i], size=(M[i], M[i+1])))
        W.append(weightMatrix)
        # biasMatrix = mat(np.full((M[i+1], 1), 1))
        biasMatrix = mat(random.uniform(-1 / M[i], 0, size=(M[i+1], 1)))
        b.append(biasMatrix)
    return W, b


def forward(M, W, b, data, label):
    """
    前向推导函数
    :param M: 层数向量
    :param W: 权值矩阵
    :param data: 传入的单个数据 (784x1矩阵)
    :param label:传入的单个标签
    :return:net, out, y, E矩阵
    """
    L = len(M)
    net = []
    out = []
    for i in range(L):
        # 初始化
        net.append(mat(np.zeros(M[i])))
        out.append(mat(np.zeros(M[i])))  # 横过来的向量
    out[0] = data  # 1x1的矩阵
    for m in range(1, L):
        if m == 1:
            net[m] = out[m - 1].transpose() * W[m - 1] + b[m - 1].transpose()  # 第m层神经元净输入，行向量
        else:
            net[m] = out[m - 1] * W[m - 1] + b[m - 1].transpose()  # 第m层神经元净输入
        out[m] = activate(net[m])  # 第m层神经元净输出
    y = net[L - 1]
    E = 0.5 * multiply(y - label, y - label)
    return net, out, y, E


def backward(M, W, b, net, out, y, E, label, rate):
    """
    后向推导函数
    :param M: 层数向量
    :param W: 权值矩阵
    :param b: 偏置矩阵
    :param net: forward return的每一层的输入,list内嵌列矩阵,列向量
    :param out: forward return的每一层的输出,list内嵌列矩阵,列向量
    :param y: forward return的该数据的预测输出,列矩阵
    :param E: forward return的预测值和真实值的平方的二分之一
    :return: 更新后的权值矩阵
    """
    grad = []
    for i in range(len(M)):
        grad.append(mat(np.zeros(M[i])).transpose())
    layer = list(range(1, len(M)))
    layer.reverse()
    for m in layer:  # 从输出层回退
        if m == len(M) - 1:  # 如果是输出层
            grad[m] = y - label
            # grad[m] = -multiply((label - y), dActivate(net[m]).transpose())
            W[m-1] -= rate * (grad[m] * out[m-1]).transpose()
            b[m-1] -= rate * grad[m]
        elif m == 1:
            t = grad[m + 1].transpose() * W[m].transpose()
            grad[m] = multiply(dActivate(net[m]).transpose(), t.transpose())
            W[m - 1] -= rate * (grad[m] * out[m - 1].transpose()).transpose()
            b[m - 1] -= rate * grad[m]
        else:
            t = grad[m+1].transpose() * W[m].transpose()
            grad[m] = multiply(dActivate(net[m]).transpose(), t.transpose())
            W[m-1] -= rate * (grad[m] * out[m-1]).transpose()
            b[m-1] -= rate * grad[m]
    return W, b


def training(M, W, b, iteration=5):
    """
    训练模型,输出结果
    :param M:神经元数量向量
    :param W: 初始化的权值矩阵
    :param iteration: 迭代次数
    :return: 训练后的结果
    """
    for iter in range(iteration):
        print("epoc %d:"%iter)
        data = mat(2 * np.pi * (np.random.random_sample() - 0.5))
        label = np.sin(data)
        net, out, y, E = forward(M, W, b, data, label)
        newRate = rate + rate/(iter + 1)
        W, b = backward(M, W, b, net, out, y, E, label, newRate)
    return W, b


def test(M, W, b):
    x = []
    predict = []
    for iter in range(100000):
        print("epoc %d:"%iter)
        # for i in range(len(dataMat)):
        randomNum = 2 * np.pi * (np.random.random_sample() - 0.5)
        data = mat(randomNum)
        label = np.sin(randomNum)
        net, out, y, E = forward(M, W, b, data, label)
        x.append(randomNum)
        predict.append(y.tolist()[0])
    plot.figure()
    plot.plot(x, predict, 'o')
    plot.show()
    return 0


def store(input, filename):
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(input, fw)
    fw.close()


def grab(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)


if __name__ == "__main__":
    M = [1, 10, 1]
    # W, b = wbInit(M)
    #
    # # dataMat = mat(dataArr)
    # # labelMat = mat(labelArr)
    # # testMat = mat(testArr)
    # # labelMat2 = mat(labelArr2)
    # # W, b = training(M, W, b, dataMat, labelMat, testMat , labelMat2 , 100)
    # W, b = training(M, W, b, 1000000)
    # store(W, 'CurveWeights.txt')
    # store(b, 'CurveBiases.txt')


    W = grab('CurveWeights.txt')
    b = grab('CurveBiases.txt')
    test(M, W, b)