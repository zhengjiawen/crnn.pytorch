import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
import cv2 as cv
import os
import numpy as np

import models.crnn as crnn


model_path = './data/crnn.pth'
gt_path = './data/res/'
img_path = '/data/home/zjw/pythonFile/masktextspotter.caffe2/lib/datasets/data/icdar2019/test_images/'
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

model = crnn.CRNN(32, 1, 37, 256)
if torch.cuda.is_available():
    model = model.cuda()
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))

converter = utils.strLabelConverter(alphabet)

transformer = dataset.resizeNormalize((100, 32))
model.eval()

# imgPathDir = os.listdir(img_path)
# for imgPath in imgPathDir:
imgPath = 'X00016469670.jpg'
imgName = imgPath.strip().split('.')[0]
img = cv.imread(img_path+imgPath)

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
        print ("pos is :" + str(pos))
        wordImg = img[minX:maxX+1, minY:maxY+1]
        image = Image.fromarray(cv.cvtColor(wordImg, cv.COLOR_BGR2RGB)).convert('L')
        image = transformer(image)
        if torch.cuda.is_available():
            image = image.cuda()
        image = image.view(1, *image.size())
        image = Variable(image)

        # model.eval()
        preds = model(image)
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)

        preds_size = Variable(torch.IntTensor([preds.size(0)]))
        sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
        raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
        print('%-20s => %-20s' % (raw_pred, sim_pred))




