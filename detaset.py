# imports and stuff
import numpy as np
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
import torch.utils.data as data
import torch.optim.lr_scheduler
import torch.nn.init
import os
from utils import *
from PIL import Image
matches = [100, 200, 300, 400, 500, 600, 700, 800]


class DF_dataset_MS(torch.utils.data.Dataset):
    def __init__(self, data_files, label_files, augmentation=True):
        super(DF_dataset_MS, self).__init__()

        self.augmentation = augmentation

        # Sanity check : raise an error if some files do not exist
        # Initialize cache dicts
        img_name_list = os.listdir(data_files)
        img_name_list.sort()
        gt_name_list = os.listdir(label_files)
        gt_name_list.sort()
        self.data_cache_ = {}
        self.label_cache_ = {}
        self.train_images_file = [os.path.join(data_files, image_name) for image_name in img_name_list]

        self.train_labels_file = [os.path.join(label_files, gt_name) for gt_name in gt_name_list]

        self.total = len(gt_name_list)

    def __len__(self):
        # Default epoch size is 10 000 samples
        return self.total

    @classmethod
    def data_augmentation(cls, data, label, flip=True, mirror=True, multi_scale=True,rotate = True,bright=True,contrast =True,):
        will_flip, will_mirror, will_multi_scale,will_rotate,will_bright,will_contrast = False,False,False,False,False,False

        if flip and random.random() < 0.5:
            will_flip = True
        if mirror and random.random() < 0.5:
            will_mirror = True
        if multi_scale and random.random() < 0.5:
            will_multi_scale = True
        if rotate and random.random() < 0.5:
            will_rotate = True
        if bright and random.random() < 0.5:
            will_bright = True
        if contrast and random.random() < 0.5:
            will_contrast = True

        if will_flip:
            label = label[::-1, :]
            data = data[:, ::-1, :]

        if will_mirror:
            label = label[:, ::-1]
            data = data[:, :, ::-1]

        if will_rotate:
            ## np to PIL
            data = np.uint8(data * 255)
            data = Image.fromarray(data.transpose((1, 2, 0)))
            label = Image.fromarray(np.uint8(label))
            alpha = 360 * (random.random() - 0.5)
            data = data.rotate(alpha)
            label = label.rotate(alpha)

            data = np.asarray(data)
            data = data.transpose((2, 0, 1)) / 255
            data = data.astype(np.float32)
            # print(data.dtype)
            label = np.asarray(label, dtype='float32')

        data_p = np.copy(data)
        label_p = np.copy(label)

        if will_multi_scale:
            scale_size = random.randint(200, 300)
            data = torch.from_numpy(np.copy(data)).unsqueeze(0)
            data = F.interpolate(data, size=(scale_size, scale_size), mode='bilinear',align_corners=True)
            data = data.squeeze().numpy()

            label = torch.from_numpy(np.copy(label)).unsqueeze(0).unsqueeze(0)
            label = F.interpolate(label, size=(scale_size, scale_size), mode='nearest')
            label = label.squeeze().numpy()

            crop_size = 256
            data_p = np.zeros((3,crop_size,crop_size),dtype='float32')
            label_p = np.zeros((crop_size, crop_size),dtype='float32')
            if scale_size>crop_size:
                x1 = random.randint(0, scale_size - crop_size)
                y1 = random.randint(0, scale_size - crop_size)
                data_p[:,:,:] = data[:,x1: x1 + crop_size,y1: y1 + crop_size]
                label_p[:,:] = label[x1: x1 + crop_size,y1: y1 + crop_size]
            else:
                data_p[:,0:scale_size,0:scale_size] = data
                label_p[0:scale_size,0:scale_size] = label

        if will_bright:
            delta = 0.1
            delta = random.uniform(-delta, delta)
            data_p += delta
            data_p = data_p.clip(min=0, max=1)

        if will_contrast:
            alpha = random.uniform(0.5, 1.5)
            data_p *= alpha
            data_p = data_p.clip(min=0, max=1)

        return data_p, label_p

    def __getitem__(self, i):
        # find the image index

        # Data is normalized in [0, 1]

        data = 1 / 255 * np.asarray(io.imread(self.train_images_file[i]).transpose((2, 0, 1)), dtype='float32')

        label = io.imread(self.train_labels_file[i])
        label = convert_from_uint16(label, matches)

        label = label.astype(np.float32)
        # print(np.unique(label))
        # Data augmentation
        if self.augmentation==True:
            data_p, label_p = self.data_augmentation(data, label)
            return (torch.from_numpy(data_p),
                    torch.from_numpy(label_p))
        else:
            return (torch.from_numpy(data),
                    torch.from_numpy(label))

