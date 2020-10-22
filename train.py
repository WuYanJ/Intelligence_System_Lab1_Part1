# coding:utf-8

import numpy as np
from PIL import Image
import os
from math import *
from numpy import *

MAX_ITER = 2
# 样本数量
# sampleAmount = 12 * 620
sampleAmount = 720
imgWidth = 28
imgHeight = 28

# 每层的神经元个数向量M，用元组记录，因为不可动态改变
M = (784, 60, 20, 12)

# 网络层数L，无需记录因为可以用len(M)得到
# L = 4
L = len(M)

# learningRate
r = 0.01


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

# 转换为灰度图像程序
input_dir = '/Users/wuyanjie/PycharmProjects/智能系统Lab1/train/1'
# input_dir = '/Users/wuyanjie/PycharmProjects/智能系统Lab1/test/1'
# input_dir = 'C:\\Users\\Tony.Hsu\\Desktop\\writingTest\\testingSet'

# out_dir = '/Users/wuyanjie/PycharmProjects/智能系统Lab1/output/1'
# out_dir = 'C:\\Users\\Tony.Hsu\\Desktop\\writingTest\\trainingSetGray'
# out_dir = 'C:\\Users\\Tony.Hsu\\Desktop\\writingTest\\testingSetGray'

listdir = os.listdir(input_dir)


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
    return im_matrix.ravel()


def weightInit():
    """
    W是一个L-1维数组，每个数组元素是一个i*j大小的matrix
    :param i:前一层的神经元数量
    :param j: 后一层的神经元数量
    :return: 初始化后的权值矩阵
    """
    for i in range(L - 1):
        # 产生一个2-8之间的，维度为（第i层神经元个数 * 第i+1层神经元个数）的随机整数矩阵
        weightMatrix = mat(random.randint(2, 8, size=(M[i], M[i + 1])))
        W.append(weightMatrix)
    return W


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


def forward(weightArray, biasArray, x, y):
    """
    一个样本的一次遍历
    :param weightArray:
    :param biasArray:
    :param x:
    :param y:
    :return:
    """
    # 这一次遍历m个样本引起的W，b的变化量（数组的每一个元素表示每一层的变化量）
    # a共L维
    # delta（z）共L-1维
    a = []
    delta = np.empty(L - 1, dtype=list)
    # delta[]和a[]都相当与局部变量，只是参与本周期的计算
    # 遍历每个样本图片
    for i in range(sampleAmount):
        dirIndex = computeDirIndex(i)
        fileIndex = computeFileIndex(i, dirIndex)
        print("./train/" + str(dirIndex) + "/" + str(fileIndex) + ".bmp")
        im_vector = img2vector("./train/" + str(dirIndex) + "/" + str(fileIndex) + ".bmp")
        a.append(im_vector)
        # 遍历每一层，求当前W，b下的各层输出a
        for layer in range(L - 1):
            a.append(activate(a[layer] * weightArray[layer]))

        # 求J对z[L]的偏导delta[L]
        # 需要考究矩阵乘法
        # delta[L] = (a[L] - y) * dActivate(mat(a[L - 1]).T * weightArray[L] + biasArray[L])

        delta[L - 2] = np.multiply(mat(a[L - 1]) - mat(y), dActivate(a[L - 2] * weightArray[L - 2]))

        # 求所有层偏导delta[l]
        for layer in range(L - 2):
            # delta[L - layer] = W[L - layer + 1].T * delta[L - layer + 1] * dActivate(mat(a[layer - 1]).T * weightArray[layer] + biasArray[layer])
            delta[L - layer - 3] = np.multiply(delta[L - layer - 2] * weightArray[L - layer - 2].T,
                                               dActivate(a[L - layer - 3] * weightArray[L - layer - 3]))
        # 更新每一层的W，b
        for layer in range(L - 1):
            # dw[L - 2 - layer] = mat(a[L - 1 - layer]).T * delta[L - 2 - layer]  # (20x1)*...(1x12)=(20x12)
            # db[L - 2 - layer] = delta[L - 2 - layer]
            dw[L - 2 - layer] = mat(a[L - 1 - layer]).T * delta[L - 2 - layer]  # (20x1)*...(1x12)=(20x12)
            db[L - 2 - layer] = delta[L - 2 - layer]
    return dw, db


def computeDirIndex(i):
    return int(i / 620 + 1)


def computeFileIndex(i, dirIndex):
    return int(i - (dirIndex - 1) * 620 + 1)


def main(W, b):
    deltaW = np.empty(L - 1, dtype=list)  # np.array内嵌list，
    deltab = np.empty(L - 1, dtype=list)
    im_vector = img2vector("./train/1/1.bmp")
    weightInit()
    for i in range(MAX_ITER):
        for image in range(sampleAmount):
            (dw1, db1) = forward(W, None, None, [1, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1])
            for layer in range(len(dw1)):
                deltaW[layer] = dw1[layer]  # +=
                deltab[layer] = db1[layer]  # +=
    #         总的deltaW和deltab得到，进入下一次iteration
        W = W - r * deltaW
        b = b - r * deltab
    print(W, b)


if __name__ == '__main__':
    main(W, b)
