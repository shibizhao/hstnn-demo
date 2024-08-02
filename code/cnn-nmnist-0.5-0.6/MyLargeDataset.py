# -*- coding: utf-8 -*-


from __future__ import print_function
import torch.utils.data as data
import torch
import numpy as np
import scipy.io as sio
import h5py
import os

# prefix = "/home/bizhao/neuro-hnn/data/N-MNIST/processed/"

class MyDataset(data.Dataset):
    def __init__(self, path='load_test.mat',method = 'h',lens = 15):
        prefix = path
        if method=='h':
            self.images = torch.from_numpy(np.load(prefix + "/" + "htrain_image.npy"))[:,:,:,:,:lens]#.float()#.cuda()
            self.labels = torch.from_numpy(np.load(prefix + "/" + "htrain_label.npy"))

        elif method=='nmnist_r':
            self.images = torch.from_numpy(np.load(prefix + "/" + "nrtest_image.npy"))[:,:,:,:,:lens]#.float().cuda()
            self.labels = torch.from_numpy(np.load(prefix + "/" + "nrtest_label.npy"))

        elif method=='nmnist_h':
            self.images = torch.from_numpy(np.load(prefix + "/" + "nhtrain_image.npy"))[:,:,:,:,:lens]#.float().cuda()
            self.labels = torch.from_numpy(np.load(prefix + "/" + "nhtrain_label.npy"))

        else:
            self.images = torch.from_numpy(np.load(prefix + "/" + "rtest_image.npy"))[:,:,:,:,:lens]#.float().cuda()
            self.labels = torch.from_numpy(np.load(prefix + "/" + "rtest_label.npy"))

        self.num_sample = int((len(self.images)//100)*100)
        print(self.images.size(),self.labels.size())

    def __getitem__(self, index): # the return value is tensor
        img, target = self.images[index], self.labels[index]
        return img, target

    def __len__(self):
        return self.num_sample
