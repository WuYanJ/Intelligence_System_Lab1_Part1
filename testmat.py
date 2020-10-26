# coding:utf-8

import numpy as np
from PIL import Image
import os
from math import *
from numpy import *
import matplotlib.pyplot as plot

if __name__ == "__main__":
    xList =[1,2,3]
    lossList = [2,4,6]
    plot.figure()
    plot.plot(xList, lossList, 'o')
    rate=1
    M=2
    iteration=0
    batch_size=99
    title='lr='+ str(rate)+' M='+str(M)+ ' iter='+str(iteration)+ ' batch_size='+ str(batch_size)
    plot.title(title)
    plot.show()