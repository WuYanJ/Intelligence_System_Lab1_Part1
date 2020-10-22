# coding:utf-8

import numpy as np
from PIL import Image
import os
from math import *
from numpy import *
import matplotlib.pyplot as plot

# 样本数量
# sampleAmount = 12 * 620
sampleAmount = 12 * 600
imgWidth = 28
imgHeight = 28

# 每层的神经元个数向量M，用元组记录，因为不可动态改变
# M = (784, 60, 20, 12)
M = (784, 60, 12)

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
        weightMatrix = mat(random.uniform(-100 / M[i], 100 / M[i], size=(M[i], M[i+1])))
        W.append(weightMatrix)
        biasMatrix = mat(random.uniform(-100 / M[i], 0, size=(M[i+1], 1)))
        b.append(biasMatrix)
    return W, b


def img2vector(filename):
    """
    :param filename: bmp文件名
    :return: 输入网络的input向量a1
    """
    # 创建零向量
    im_matrix = np.zeros((1, imgWidth * imgHeight))
    # 打开文件
    im = Image.open(filename)
    # print(im.format, im.size, im.mode)
    # 图像矩阵
    im_matrix = np.array(im)

    # 拉成一维向量
    return mat(im_matrix.ravel()).transpose()  # 列向量


def computeDirIndex(i):
    # return int(i / 620 + 1)
    # return int(i / 600 + 1)
    return int(i / 20 + 1)


def computeFileIndex(i, dirIndex):
    # return int(i - (dirIndex - 1) * 620 + 1)
    # return int(i - (dirIndex - 1) * 600 + 1)
    return int(i - (dirIndex - 1) * 20 + 1)


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
    out[0] = data
    y = mat(np.zeros(12)).T
    E = mat(np.zeros(12)).T
    for m in range(1, L):
        if m == 1:
            net[m] = out[m - 1].transpose() * W[m - 1] + b[m - 1].transpose()  # 第m层神经元净输入
        else:
            net[m] = out[m - 1] * W[m - 1] + b[m - 1].transpose()  # 第m层神经元净输入
        out[m] = activate(net[m])  # 第m层神经元净输出
    y = out[L - 1].transpose()
    # E = 0.5 * multiply(y - label, y - label)
    E = 0.5 * (y-label).T * (y-label)
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
            grad[m] = -multiply((label - y), dActivate(net[m]).transpose())
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
    lossList=[]
    xList=[]
    for iter in range(iteration):
        print("epoc %d:"%iter)
        # for i in range(len(dataMat)):
        sampleList = list(range(sampleAmount))
        random.shuffle(sampleList)
        # for i in range(sampleAmount):
        loss = 0
        for i in sampleList:
            dirIndex = computeDirIndex(i)
            fileIndex = computeFileIndex(i, dirIndex)
            label = [0] * 12
            label[(int)(dirIndex-1)] = 1
            # print("label:", label)
            print("./train/" + str(dirIndex) + "/" + str(fileIndex) + ".bmp")
            im_matrix = img2vector("./train/" + str(dirIndex) + "/" + str(fileIndex) + ".bmp")  # 列向量
            lab_matrix = mat(label).transpose()
            net, out, y, E = forward(M, W, b, im_matrix, lab_matrix)
            # newRate = rate + 1/(iter + 1)
            newRate = rate + rate/(iter + 1)
            (W, b) = backward(M, W, b, net, out, y, E, lab_matrix, newRate)
            loss+=E[0]
        xList.append(iter+1)
        lossList.append(loss.tolist()[0])
    plot.figure()
    plot.plot(xList, lossList, 'o')
    plot.show()
    # print("第%d个样本训练！"%i)
    # error = test(dataMat , labelMat , M , W , b)
    # error2 = test(testMat , labelMat2 , M , W , b)
    return W, b


def test(M, W, b):
    count = 0
    # for i in range(sampleAmount, sampleAmount+100):
    for i in range(12 * 20):
        dirIndex = computeDirIndex(i)
        fileIndex = computeFileIndex(i, dirIndex)+600
        label = [0] * 12
        label[(int)(dirIndex - 1)] = 1
        # print("label:", label)
        print("./temp/" + str(dirIndex) + "/" + str(fileIndex) + ".bmp")
        # print("./train/" + str(dirIndex) + "/" + str(fileIndex) + ".bmp")
        # im_matrix = img2vector("./train/" + str(dirIndex) + "/" + str(fileIndex) + ".bmp")  # 列向量
        im_matrix = img2vector("./temp/" + str(dirIndex) + "/" + str(fileIndex) + ".bmp")  # 列向量
        lab_matrix = mat(label).transpose()

        net, out, y, E = forward(M, W, b, im_matrix, lab_matrix)
        print("y ", y)
        t = argmax(y)
        if t == argmax(lab_matrix):
            count += 1
            print("data%d 测试正确\n" % i)
    print("正确率：%f" % (count/240))
    return count/240


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
    # M = [784, 60, 12]
    # W, b = wbInit(M)
    #
    # # dataMat = mat(dataArr)
    # # labelMat = mat(labelArr)
    # # testMat = mat(testArr)
    # # labelMat2 = mat(labelArr2)
    # # W, b = training(M, W, b, dataMat, labelMat, testMat , labelMat2 , 100)
    # W, b = training(M, W, b, 100)
    # store(W, 'weights.txt')
    # store(b, 'biases.txt')
    W = grab('weights.txt')
    b = grab('biases.txt')
    test(M, W, b)