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
sampleAmount = 12 * 2600

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
        weightMatrix = mat(np.zeros((M[i], M[i+1])))
        W.append(weightMatrix)
        biasMatrix = mat(np.zeros((M[i+1], 1)))
        b.append(biasMatrix)
    return W, b


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


def computeDirIndex(i):
    # return int(i / 600 + 1)
    return int(i / 2600 + 1)
    # return int(i / 20 + 1)


def computeFileIndex(i, dirIndex):
    # return int(i - (dirIndex - 1) * 600 + 1)
    return int(i - (dirIndex - 1) * 2600 + 1)
    # return int(i - (dirIndex - 1) * 20 + 1)

def computeTestDirIndex(i):
    # return int(i / 600 + 1)
    # return int(i / 600 + 1)
    return int(i / 420 + 1)


def computeTestFileIndex(i, dirIndex):
    # return int(i - (dirIndex - 1) * 620 + 1)
    # return int(i - (dirIndex - 1) * 600 + 1)
    return int(i - (dirIndex - 1) * 420 + 1)


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
    E = (-1) * log(y.tolist()[l])  # -ln(yi)
    return net, out, y, E


def backward(M, W, b, net, out, y, E, label, rate, lmd=0.00015):
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
    for i in range(len(M)):
        grad.append(mat(np.zeros(M[i])).transpose())  # grad的每一个列表项都是一个列向量
    layer = list(range(1, len(M)))
    layer.reverse()
    for m in layer:  # 从输出层回退
        if m == len(M) - 1:  # 如果是输出层
            grad[m] = -(label - y)
            W[m - 1] -= (rate * (grad[m] * out[m - 1]).transpose() + rate * lmd * W[m - 1])
            b[m - 1] -= rate * grad[m]
        elif m == 1:
            t = grad[m + 1].transpose() * W[m].transpose()
            grad[m] = multiply(dActivate(net[m]).transpose(), t.transpose())
            W[m - 1] -= (rate * (grad[m] * out[m - 1].transpose()).transpose() + rate * lmd * W[m - 1])
            b[m - 1] -= rate * grad[m]
        else:
            t = grad[m + 1].transpose() * W[m].transpose()
            grad[m] = multiply(dActivate(net[m]).transpose(), t.transpose())
            W[m - 1] -= (rate * (grad[m] * out[m - 1]).transpose() + rate * lmd * W[m - 1])
            b[m - 1] -= rate * grad[m]
    return W, b


def training(M, W, b, iteration=5):
    """
    训练模型,输出结果
    :param M:神经元数量向量
    :param W: 初始化的权值矩阵
    :param iteration: 迭代次数
    :return: 训练后的结果
    """
    lossList = []
    testLossList = []
    xList = []
    for iter in range(iteration):
        print("epoc %d:" % iter)
        test(M, W, b)
        sampleList = list(range(sampleAmount))
        random.shuffle(sampleList)
        loss = 0
        train_count = sampleAmount
        train_correct = 0
        for i in sampleList:
            dirIndex = computeDirIndex(i)
            fileIndex = computeFileIndex(i, dirIndex)
            label = [0] * 12
            label[(int)(dirIndex - 1)] = 1
            # print("./train/" + str(dirIndex) + "/" + str(fileIndex) + ".bmp")
            if fileIndex > 600:
                filepath = "./test/" + str(dirIndex) + "/" + str(fileIndex) + ".bmp"
            else:
                filepath = "./train/" + str(dirIndex) + "/" + str(fileIndex) + ".bmp"
            im_matrix = img2vector(filepath)  # 列向量
            if im_matrix is None:
                train_count -= 1
                # print(filepath, " not exist")
                continue
            lab_matrix = mat(label).transpose()
            net, out, y, E = forward(M, W, b, im_matrix, lab_matrix)
            t = argmax(y)
            if t == argmax(lab_matrix):
                train_correct += 1
            newRate = rate + rate / (iter + 1)
            (W, b) = backward(M, W, b, net, out, y, E, lab_matrix, newRate)
            # loss += E[0]
            loss += E
        xList.append(iter + 1)
        lossList.append(loss.tolist()[0])
        print("Train正确率：", train_correct / train_count)
        print(loss.tolist()[0], "\n")
        # lossList.append(loss)


        lossTest=0
        for i in range(12 * 420):
            dirIndex = computeTestDirIndex(i)
            fileIndex = computeTestFileIndex(i, dirIndex) + 600
            label = [0] * 12
            label[(int)(dirIndex - 1)] = 1
            # print("./temp/" + str(dirIndex) + "/" + str(fileIndex) + ".bmp")
            # print("./train/" + str(dirIndex) + "/" + str(fileIndex) + ".bmp")
            # im_matrix = img2vector("./train/" + str(dirIndex) + "/" + str(fileIndex) + ".bmp")  # 列向量
            filepath = "./temp/" + str(dirIndex) + "/" + str(fileIndex) + ".bmp"
            im_matrix = img2vector(filepath)  # 列向量
            if im_matrix is None:
                # print(filepath, " not exist")
                continue
            lab_matrix = mat(label).transpose()
            net, out, y, E = forward(M, W, b, im_matrix, lab_matrix)
            lossTest += E
        testLossList.append(lossTest.tolist()[0] * 6)
    plot.figure()
    plot.plot(xList, lossList, 'o')
    plot.plot(xList, testLossList, 'ro')
    title = 'lr=' + str(rate) + ' M=' + str(M) + ' iter=' + str(iteration) + ' Winit=[-0.1,0.1] CE lmd=0.00014'
    plot.title(title)
    plot.show()
            # print("第%d个样本训练！"%i)
        # error = test(dataMat , labelMat , M , W , b)
        # error2 = test(testMat , labelMat2 , M , W , b)
    return W, b


def test(M, W, b):
    count = 0
    sum = 5040
    # for i in range(sampleAmount, sampleAmount+100):
    for i in range(12 * 420):
        dirIndex = computeTestDirIndex(i)
        fileIndex = computeTestFileIndex(i, dirIndex) + 600
        label = [0] * 12
        label[(int)(dirIndex - 1)] = 1
        # print("./temp/" + str(dirIndex) + "/" + str(fileIndex) + ".bmp")
        # print("./train/" + str(dirIndex) + "/" + str(fileIndex) + ".bmp")
        # im_matrix = img2vector("./train/" + str(dirIndex) + "/" + str(fileIndex) + ".bmp")  # 列向量
        filepath = "./temp/" + str(dirIndex) + "/" + str(fileIndex) + ".bmp"
        im_matrix = img2vector(filepath)  # 列向量
        if im_matrix is None:
            sum -= 1
            # print(filepath, " not exist")
            continue
        lab_matrix = mat(label).transpose()
        # print("./temp/" + str(dirIndex) + "/" + str(fileIndex) + ".bmp")
        net, out, y, E = forward(M, W, b, im_matrix, lab_matrix)
        # print("y：", y)
        t = argmax(y)
        if t == argmax(lab_matrix):
            count += 1
            # print("data%d 测试正确\n" % i)
    print("正确率：%f" % (count / sum))
    return count / sum


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
    M = [784, 60, 12]
    W, b = wbInit(M)
    W, b = training(M, W, b, 60)
    store(W, 'Params/weightstest.txt')
    store(b, 'Params/biasestest.txt')

    # W = readParam('Params/weightstest.txt')
    # b = readParam('Params/biasestest.txt')
    # test(M, W, b)