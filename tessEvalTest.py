import tesserocr
import cv2 as cv
from PIL import Image
import os
import numpy as np

debug = True

def regWordByTesserocr(img):
    image = Image.fromarray(img)
    result = tesserocr.image_to_text(image, lang='chi_sim+equ+eng')
    return result

gt_path = './data/res/'

img_path = '/data/home/zjw/pythonFile/masktextspotter.caffe2/lib/datasets/data/icdar2019/test_images/'
imgPath = 'X00016469670.jpg'
imgName = imgPath.strip().split('.')[0]
img = cv.imread(img_path+imgPath)
if debug:
    print ("img shape: "+str(img.shape))
    print (len(img.shape))

with open(gt_path+'res_'+imgName+'.txt') as f:
    lines = f.readlines()
    for line in lines:
        pos = line.split(',')
        pos = np.array(pos, dtype=np.int)
        pos = pos.reshape((-1,2))
        minX = min(pos[:,0])
        minY = min(pos[:,1])
        maxX = max(pos[:,0])
        maxY = max(pos[:,1])
        if debug:
            print ("pos is :" + str(pos))
        wordImg = img[minX:maxX+1, minY:maxY+1,:]
        print ("word img shape: "+str(wordImg.shape))
        if len(img.shape) == 3:
            wordImg = cv.cvtColor(wordImg, cv.COLOR_BGR2RGB)
        result = regWordByTesserocr(wordImg)
        print("word :"+result)



