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

# 每层的神经元个数向量M，用元组记录，因为不可动态改变
# M = (784, 30, 12)
M = (784, 1000, 12)

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


# 损失函数选择
def lossFunc(output, label):
    return 0


# def dloss(x, label):
#     if type(x) != int:
#         d = np.zeros(shape(x))
#         for i in range(len(x)):
#             d[i] = label[i] / x[i]
#     else:
#         d = 1 / x
#     return d


# 权重W
W = []
# 偏置b
b = []

# 用来存放一次迭代中，每一个样本图像得出的梯度值
# dw = np.empty(L - 1, dtype=list)
# db = np.empty(L - 1, dtype=list)


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
        # print(y[i],",")
    return y


def wbInit(M):
    for i in range(len(M) - 1):
        weightMatrix = mat(random.uniform(-100/M[i], 100/M[i], size=(M[i], M[i + 1])))
        W.append(weightMatrix)
        # if i == len(M) - 2:
        #     biasMatrix = mat(random.uniform(-0.2, 0.2, size=(M[i + 1], 1)))
        # else:
        #     biasMatrix = mat(random.uniform(-1, 0, size=(M[i + 1], 1)))
        biasMatrix = mat(random.uniform(-100 / M[i], 0, size=(M[i+1], 1)))
        b.append(biasMatrix)
    return W, b


def img2vector(filename):
    """
    :param filename: bmp文件名
    :return: 输入网络的input向量a1
    """
    # 打开文件
    im = Image.open(filename)
    # print(im.format, im.size, im.mode)
    # 图像矩阵
    im_matrix = np.array(im)

    # 拉成一维向量
    return mat(im_matrix.ravel()).transpose()  # 列向量


def computeDirIndex(i):
    # return int(i / 620 + 1)
    return int(i / 600 + 1)
    # return int(i / 20 + 1)


def computeFileIndex(i, dirIndex):
    # return int(i - (dirIndex - 1) * 620 + 1)
    return int(i - (dirIndex - 1) * 600 + 1)
    # return int(i - (dirIndex - 1) * 20 + 1)

def computeTestDirIndex(i):
    # return int(i / 620 + 1)
    # return int(i / 600 + 1)
    return int(i / 20 + 1)


def computeTestFileIndex(i, dirIndex):
    # return int(i - (dirIndex - 1) * 620 + 1)
    # return int(i - (dirIndex - 1) * 600 + 1)
    return int(i - (dirIndex - 1) * 20 + 1)


def matrix_add(matrix1, matrix2):
    total_element = [matrix1[i][j] + matrix2[i][j] for i in range(len(matrix1)) for j in range(len(matrix1))]
    new_matrix = [total_element[x:x+len(matrix1)] for x in range(0,len(total_element),len(matrix1))]
    return new_matrix


def forward(M, W, b, data, label):
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
    for i in range(12):
        if label[i] == 1:
            l=i
            break
    Emse = 0.5 * (y - label).T * (y - label)
    E = (-1) * log(y.tolist()[l])  # -ln(yi)
    return net, out, y, E, Emse


# backward函数并不计算更新后的W和b，而是计算每一个样本输入后W,b的梯度值（即变化量），在backward外的某个变量收集这些输出的梯度值
def backward(M, W, b, net, out, y, E, label, rate):
    """
    后向推导函数，返回更新后的W, b参数
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
    lmd = 0.1
    for i in range(len(M)):
        grad.append(mat(np.zeros(M[i])).transpose())  # grad的每一个列表项都是一个列向量
    dW = []
    db = []
    for i in range(len(M) - 1):
        dW.append(mat(np.zeros((M[i], M[i + 1]))))  # 每一个列表项都是784x60,60x12的矩阵
        db.append(mat(np.zeros((M[i + 1], 1))))
    layer = list(range(1, len(M)))
    layer.reverse()
    for m in layer:  # 从输出层回退
        if m == len(M) - 1:  # 如果是输出层
            # grad[m] = -(label - y)
            grad[m] = -(label - y)
            dW[m - 1] = (grad[m] * out[m - 1]).transpose()
            db[m - 1] = grad[m]
        elif m == 1:
            t = grad[m + 1].transpose() * W[m].transpose()
            grad[m] = multiply(dActivate(net[m]).transpose(), t.transpose())
            dW[m - 1] = (grad[m] * out[m - 1].transpose()).transpose()
            db[m - 1] = grad[m]
        else:
            t = grad[m + 1].transpose() * W[m].transpose()
            grad[m] = multiply(dActivate(net[m]).transpose(), t.transpose())
            dW[m - 1] = (grad[m] * out[m - 1]).transpose()
            db[m - 1] = grad[m]
    return dW, db


def training(M, W, b, iteration):
    """
    训练模型,输出结果
    :param M:神经元数量向量
    :param W: 初始化的权值矩阵
    :param iteration: 迭代次数
    :return: 训练后的结果
    """
    lossList = []
    lossmselist=[]
    xList = []
    for iter in range(iteration):
        print("epoc %d:" % iter)
        sampleList = list(range(sampleAmount))
        random.shuffle(sampleList)
        loss=0
        lossmse=0
        batch_size = 12
        batch_amount = (int)(sampleAmount / batch_size)
        start=0
        for batch in range(batch_amount):
            DW = []
            Db = []
            for i in range(len(M) - 1):
                DW.append(mat(np.zeros((M[i], M[i + 1]))))  # 每一个列表项都是784x60,60x12的矩阵
                Db.append(mat(np.zeros((M[i + 1], 1))))
            for i in sampleList[start:start+12:1]:
                dirIndex = computeDirIndex(i)
                fileIndex = computeFileIndex(i, dirIndex)
                label = [0] * 12
                label[(int)(dirIndex - 1)] = 1
                # print("./train/" + str(dirIndex) + "/" + str(fileIndex) + ".bmp")
                im_matrix = img2vector("./train/" + str(dirIndex) + "/" + str(fileIndex) + ".bmp")  # 列向量
                lab_matrix = mat(label).transpose()
                net, out, y, E, Emse = forward(M, W, b, im_matrix, lab_matrix)
                newRate = rate + rate / (iter + 1)
                dW, db = backward(M, W, b, net, out, y, E, lab_matrix, newRate)
                #  把这一次迭代的dW和db加进累积的DW和Db中去
                for j in range(len(DW)):
                    DW[j] += dW[j]
                    Db[j] += db[j]
                loss += E  # 每一个sample都有一个loss
                lossmse += Emse
            for m in range(len(W)):  # 对于每一个权重矩阵
                W[m] -= (rate / batch_size) * DW[m]
                # print("W[1]\n", W[1])
                b[m] -= (rate / batch_size) * Db[m]
        print(loss.tolist()[0])
        print(lossmse.tolist()[0])
        # if iter > 190:
        #     store(W, 'weights'+str(iter)+"-test.txt")
        #     store(b, 'biases'+str(iter)+"-test.txt")
        xList.append(iter + 1)
        lossList.append(loss.tolist()[0])
        lossmselist.append(lossmse.tolist()[0])
    plot.figure()
    plot.plot(xList, lossList, 'o')
    plot.plot(xList, lossmselist, 'ro')
    title = 'lr=' + str(rate) + ' M=' + str(M) + ' iter=' + str(iteration) + ' batch_size=' + str(batch_size) + ' CE'
    plot.title(title)
    plot.show()
    return W, b


def test(M, W, b):
    count = 0
    # for i in range(sampleAmount, sampleAmount+100):
    for i in range(12 * 20):
        dirIndex = computeDirIndex(i)
        fileIndex = computeFileIndex(i, dirIndex) + 600
        label = [0] * 12
        label[(int)(dirIndex - 1)] = 1
        print("./temp/" + str(dirIndex) + "/" + str(fileIndex) + ".bmp")
        # print("./train/" + str(dirIndex) + "/" + str(fileIndex) + ".bmp")
        # im_matrix = img2vector("./train/" + str(dirIndex) + "/" + str(fileIndex) + ".bmp")  # 列向量
        im_matrix = img2vector("./temp/" + str(dirIndex) + "/" + str(fileIndex) + ".bmp")  # 列向量
        lab_matrix = mat(label).transpose()
        net, out, y, E, Emse = forward(M, W, b, im_matrix, lab_matrix)
        print("y：", y)
        t = argmax(y)
        if t == argmax(lab_matrix):
            count += 1
            print("data%d 测试正确\n" % i)
    print("正确率：%f" % (count / 240))
    return count / 240


def store(input, filename):
    import pickle
    fw = open(filename, 'wb')
    pickle.dump(input, fw)
    fw.close()


def readParam(filename):
    import pickle
    fr = open(filename, 'rb')
    return pickle.load(fr)


if __name__ == "__main__":
    M = [784, 1000, 12]
    W, b = wbInit(M)
    W, b = training(M, W, b, 100)
    store(W, './Params/weights.txt')
    store(b, './Params/biases.txt')
    # W = readParam('./Params/weights.txt')
    # b = readParam('./Params/biases.txt')
    # print(W)
    # W = readParam('./Params/weights0.85MSE.txt')
    # b = readParam('./Params/biases0.85MSE.txt')
    # W, b = training(M, W, b, 50)
    # store(W, './Params/weights-batch0.77.txt')
    # store(b, './Params/biases-batch0.77.txt')
    # W = readParam('./Params/weights.txt')
    # b = readParam('./Params/biases.txt')
    # test(M, W, b)