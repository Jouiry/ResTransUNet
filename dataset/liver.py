import numpy as np
import cv2  #https://www.jianshu.com/p/f2e88197e81d
import random

from skimage.io import imread
from skimage import color

import torch
import torch.utils.data
from torchvision import datasets, models, transforms


class Dataset_liver(torch.utils.data.Dataset):

    def __init__(self, args, img_paths, mask_paths, transform):
        self.args = args
        self.img_paths = img_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        mask_path = self.mask_paths[idx]
        #读numpy数据(npy)的代码
        npimage = np.load(img_path)
        npmask = np.load(mask_path)

        # npimage = npimage.transpose((2, 0, 1))

        liver_label = npmask.copy()
        liver_label[npmask == 2] = 1
        liver_label[npmask == 1] = 1



        nplabel = np.empty((448,448,1))
        nplabel[:, :, 0] = liver_label
        # nplabel = nplabel.transpose((2, 0, 1))

        nplabel = nplabel.astype("float32")
        npimage = npimage.astype("float32")


        if self.transform:
            p1 = random.randint(0, 1)
            p2 = random.randint(0, 1)
            trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(224),
                transforms.RandomHorizontalFlip(p1),
                transforms.RandomVerticalFlip(p2),
                # transforms.RandomRotation(50),

            ])
            seed = np.random.randint(2147483647)
            random.seed(seed)
            npimage = trans(npimage)
            random.seed(seed)
            nplabel = trans(nplabel)
        else:
            trans = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize(224),
            ])
            seed = np.random.randint(2147483647)
            random.seed(seed)
            npimage = trans(npimage)
            random.seed(seed)
            nplabel = trans(nplabel)


        return npimage,nplabel
