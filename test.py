# coding:utf-8

import numpy as np
from PIL import Image
import os
from math import *
from numpy import *
import matplotlib.pyplot as plot

# 样本数量
# sampleAmount = 12 * 620
# sampleAmount = 12 * 600
# sampleAmount = 12 * 2600

# 每层的神经元个数向量M，用元组记录，因为不可动态改变
# M = (784, 200, 60, 12)
# M = (784, 20, 12)

imgWidth=28
imgHeight=28
# 网络层数len(M)
# L = len(M)

# learningRate
rate = 0.01


# 激活函数选择
def activate(x):
    return sigmoid(x)


def dActivate(x):
    return dSigmoid(x)


# 权重W
W = []
# 偏置b
b = []


# sigmoid函数
def sigmoid(x):
    if type(x) != int:
        y = np.zeros(shape(x))
        for i in range(len(x)):
            if x.tolist()[0][i] > 0:
                y[i] = 1 / (1 + exp((-1) * x[i]))
            else:
                y[i] = exp(x[i]) / (1 + exp(x[i]))
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


def softmax(x):
    denominator = 0  # 求分母sigma exp
    for i in range(len(x)):
        denominator += exp(x[i])
    y = np.zeros(shape(x))
    for i in range(len(x)):
        y[i] = exp(x[i]) / denominator
    return y


def img2vector(filename):
    """
    :param filename: bmp文件名
    :return: 输入网络的input向量a1
    """
    if not os.path.exists(filename):
        return None
    # 打开文件
    im = Image.open(filename)
    # print(im.format, im.size, im.mode)
    # 图像矩阵
    im_matrix = np.array(im)

    # 拉成一维向量
    return mat(im_matrix.ravel()).transpose()  # 列向量


def forward(M, W, b, data):
    """
    前向推导函数，返回每层的net, out
    :param M: 层数向量
    :param W: 权值矩阵
    :param data: 传入的单个数据 (784x1矩阵)
    :param label: 传入的单个标签
    :return: net, out, y, E矩阵
    """
    L = len(M)
    net = []
    out = []
    for i in range(L):
        # 初始化
        net.append(mat(np.zeros(M[i])))
        out.append(mat(np.zeros(M[i])))  # 横过来的向量
    out[0] = data
    for m in range(1, L):
        if m == 1:
            net[m] = out[m - 1].transpose() * W[m - 1] + b[m - 1].transpose()  # 第m层神经元净输入
        else:
            net[m] = out[m - 1] * W[m - 1] + b[m - 1].transpose()  # 第m层神经元净输入
        out[m] = activate(net[m])  # 第m层神经元净输出
    out[L - 1] = mat(softmax(net[L - 1].T.tolist())).T
    y = out[L - 1].transpose()  # fy是横过来的向量，是softmax以前的f(x）
    return net, out, y


def test(M, W, b):
    predictFile = open("pred.txt", "w")
    # for i in range(sampleAmount, sampleAmount+100):
    for i in range(1801):
        filepath = "./test/" + str(i) + ".bmp"
        im_matrix = img2vector(filepath)  # 列向量
        if im_matrix is None:
            # print(filepath, " not exist")
            continue
        net, out, y = forward(M, W, b, im_matrix)
        # print("y：", y)
        t = argmax(y)
        predictFile.write(str(t+1))
        predictFile.write("\n")
        print(t + 1)
    predictFile.close()
    return t


def readParam(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)


if __name__ == "__main__":
    M = [784, 60, 12]
    W = readParam('Params/weightsFinal.txt')
    b = readParam('Params/biasesFinal.txt')
    test(M, W, b)