# coding:utf-8
import time
from glob import glob

import numpy as np
from PIL import Image

import model
# ces

paths = glob('./test/*.*')

if __name__ == '__main__':
    im = Image.open("./pic/image01.jpg")
    img = np.array(im.convert('RGB'))
    t = time.time()
    '''
    result,img,angel分别对应-识别结果，图像的数组，文字旋转角度
    '''
    result, img, angle = model.model(
        img, model='pytorch', adjust=True, detectAngle=True)
    print("It takes time:{}s".format(time.time() - t))
    print("---------------------------------------")
    print('result:',result)
    print("---------------------------------------")
    for key in result:
        print(result[key][1])
