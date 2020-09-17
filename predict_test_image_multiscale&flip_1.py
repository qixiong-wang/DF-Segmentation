from skimage import io
from glob import glob
from tqdm import tqdm_notebook as tqdm
from sklearn.metrics import confusion_matrix
import random
import itertools
# Matplotlib
import matplotlib.pyplot as plt
# Torch imports
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.optim as optim
import torch.optim.lr_scheduler
import torch.nn.init
from torch.autograd import Variable
from torch.backends import cudnn
import cv2
import os
from PIL import Image
from resnet import resnet101
from torch.nn.functional import interpolate,normalize
from torch.nn import Module, Sequential, Conv2d, ReLU,AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from utils import *
from model import DANet
from detaset import DF_dataset

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

DATA_FOLDER = '../image_A'
img_name_list = os.listdir(DATA_FOLDER)
img_name_list.sort()
N_CLASSES=8
LABELS =['water','Transportation','building','cultivated area','grassland','woodland','Soil','others']
matches = [100, 200, 300, 400, 500, 600, 700, 800]


net=torch.nn.DataParallel(DANet(nclass=8, backbone='resnet101'), device_ids=[0])
net.cuda()

net.load_state_dict(torch.load('./DA_DF_baseline_3*3_epoch50'))

PRED_FOLDER = "./results"
if not os.path.exists(PRED_FOLDER):
    os.makedirs(PRED_FOLDER)
scales = [256]

def save_img(net,DATA_FOLDER,PRED_FOLDER):

    # Use the network on the test set

    # Switch the network to inference mode
    net.eval()
    with torch.no_grad():
        for img_name in img_name_list[50000:]:
            test_image = 1 / 255 * np.asarray(io.imread(os.path.join(DATA_FOLDER,img_name)).transpose((2, 0, 1)), dtype='float32')
            test_image_flip = np.copy(test_image[:, ::-1, :])
            test_image_mirror = np.copy(test_image[:, :, ::-1])
            test_image_flip_mirror = np.copy(test_image_flip[:, :, ::-1])

            test_image = torch.from_numpy(test_image).cuda().unsqueeze(0)
            test_image_flip = torch.from_numpy(test_image_flip).cuda().unsqueeze(0)
            test_image_mirror = torch.from_numpy(test_image_mirror).cuda().unsqueeze(0)
            test_image_flip_mirror = torch.from_numpy(test_image_flip_mirror).cuda().unsqueeze(0)

            outs = np.zeros((8,256,256),dtype=np.float)

            for scale in scales:
                temp = F.interpolate(test_image, size=(scale, scale), mode='bilinear')
                temp_flip = F.interpolate(test_image_flip, size=(scale, scale), mode='bilinear')
                temp_mirror = F.interpolate(test_image_mirror, size=(scale, scale), mode='bilinear')
                temp_flip_mirror = F.interpolate(test_image_flip_mirror, size=(scale, scale), mode='bilinear')

                temp_out = net(temp)
                temp_out = F.softmax(temp_out[0], dim=1)
                temp_out = F.interpolate(temp_out, size=(256, 256), mode='bilinear').cpu().numpy()

                temp_flip_out = net(temp_flip)
                temp_flip_out = F.softmax(temp_flip_out[0], dim=1)
                temp_flip_out = F.interpolate(temp_flip_out, size=(256, 256), mode='bilinear').cpu().numpy()
                temp_flip_out = np.copy(temp_flip_out[:, :, ::-1, :])

                temp_mirror_out = net(temp_mirror)
                temp_mirror_out = F.softmax(temp_mirror_out[0], dim=1)
                temp_mirror_out = F.interpolate(temp_mirror_out, size=(256, 256), mode='bilinear').cpu().numpy()
                temp_mirror_out = np.copy(temp_mirror_out[:, :, :, ::-1])

                temp_flip_mirror_out = net(temp_flip_mirror)
                temp_flip_mirror_out = F.softmax(temp_flip_mirror_out[0], dim=1)
                temp_flip_mirror_out = F.interpolate(temp_flip_mirror_out, size=(256, 256), mode='bilinear').cpu().numpy()
                temp_flip_mirror_out = np.copy(temp_flip_mirror_out[:, :, ::-1, ::-1])

                outs =outs +temp_out +temp_flip_out +temp_mirror_out +temp_flip_mirror_out


            pred = np.argmax(outs, axis=1) ## 1*H*W
            pred = np.squeeze(pred,axis=0)


            saved_path = os.path.join(PRED_FOLDER,img_name.split('.')[0]+'.png')
            pred_img = convert_to_uint16(pred,matches)
            saved_img = Image.fromarray(pred_img)
            saved_img.save(saved_path, format='PNG')


save_img(net,DATA_FOLDER,PRED_FOLDER)