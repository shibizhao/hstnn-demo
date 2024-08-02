import argparse
import time
import math
import os
import copy
import sys
import torch
import torch.onnx
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import logging


smnist_dataset_path = "./MNIST/"
traindataset = torchvision.datasets.MNIST(root=smnist_dataset_path, train=True, download=True, transform=transforms.ToTensor())
#trainloader  = torch.utils.data.DataLoader(traindataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

testset      = torchvision.datasets.MNIST(root=smnist_dataset_path, train=False, download=True,  transform=transforms.ToTensor())
#testloader   = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)
