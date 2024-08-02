import torch
import numpy as np
import scipy.io as sio
import h5py
import os
import sys

torch.multiprocessing.set_start_method('spawn')
torch.set_num_threads(16)

# be sure the current path is `hstnn/data` and have downloaded the matrix data in `hstnn/data/N-MNIST`
def nmnist_numpy_gen(dt=3):

    folder = "./N-MNIST/processed/" + str(dt) + "ms" + "/"
    os.system("mkdir -p " + folder)
    # train matrix
    train_mat_name = "./N-MNIST/NMNIST_train_data_" + str(dt) + "ms.mat"
    data = h5py.File(train_mat_name)
    image,label = data['image'],data['label']
    image = np.transpose(image)
    label = np.transpose(label)

    image = torch.from_numpy(image)
    label = torch.from_numpy(label).float()
    np.save(folder + "htrain_image.npy", image)
    np.save(folder + "htrain_label.npy", label)
    
    image = image.permute(0, 3, 1, 2, 4)
    np.save(folder + "nhtrain_image.npy", image)
    np.save(folder + "nhtrain_label.npy", label)

    # test matrix

    test_mat_name = "./N-MNIST/NMNIST_test_data_" + str(dt) + "ms.mat"
    data = sio.loadmat(test_mat_name)

    image = torch.from_numpy(data['image'])
    label = torch.from_numpy(data['label']).float()
    np.save(folder + "rtest_image.npy", image)
    np.save(folder + "rtest_label.npy", label)

    image = image.permute(0, 3, 1, 2, 4)
    np.save(folder + "nrtest_image.npy", image)
    np.save(folder + "nrtest_label.npy", label)    


# def nmnist_compare(dt=3):
#     c1 = torch.sum(torch.abs(torch.from_numpy(np.load("./htrain_image.npy")) - torch.from_numpy(np.load("./N-MNIST/processed-bak/htrain_image.npy"))))
#     c2 = torch.sum(torch.abs(torch.from_numpy(np.load("./nhtrain_image.npy")) - torch.from_numpy(np.load("./N-MNIST/processed-bak/nhtrain_image.npy"))))
#     c3 = torch.sum(torch.abs(torch.from_numpy(np.load("./rtest_image.npy")) - torch.from_numpy(np.load("./N-MNIST/processed-bak/rtest_image.npy"))))
#     c4 = torch.sum(torch.abs(torch.from_numpy(np.load("./nrtest_image.npy")) - torch.from_numpy(np.load("./N-MNIST/processed-bak/nrtest_image.npy"))))
#     print(c1, c2, c3, c4)
# nmnist_compare(dt=3)

try: 
    dt = int(sys.argv[1])
except:
    dt = 3
print("dt: ", dt, "ms")

nmnist_numpy_gen(dt = dt)
