import os
from PIL import Image
from PIL import ImageOps


def computeDirIndex(i):
    # return int(i / 620 + 1)
    return int(i / 600 + 1)
    # return int(i / 20 + 1)


def computeFileIndex(i, dirIndex):
    # return int(i - (dirIndex - 1) * 620 + 1)
    return int(i - (dirIndex - 1) * 600 + 1)
    # return int(i - (dirIndex - 1) * 20 + 1)

for i in range(12):
    filepath="./test/" + str(i+1)
    os.mkdir(filepath)

for i in range(600*12):  # 每个i对应16张增强图片
    dirIndex = computeDirIndex(i)
    fileIndex = computeFileIndex(i, dirIndex)
    filename = "./train/" + str(dirIndex) + "/" + str(fileIndex) + ".bmp"
    orImage = Image.open(filename)
    image = ImageOps.expand(orImage, (1, 1, 1, 1), fill='white')
    for x in range(2):
        for y in range(2):
            cropImage = ImageOps.crop(image, (x, y, 2 - x, 2 - y))
            newname = 600 + (fileIndex - 1) * 4 + x * 2 + y + 1
            cropImage.save('./test/' + str(dirIndex) + "/" + str(newname) + ".bmp")
