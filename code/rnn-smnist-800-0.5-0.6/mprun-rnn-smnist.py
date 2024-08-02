import os
import subprocess

path = "./"

gpus = 2

available = [i for i in range(gpus)]
seed_list = [1111 * i for i in range(1, 5)]

cnt = 0

def multi_task(path, parameters):
    global cnt
    for r in parameters["ratio"]:
        for se in parameters["seed"]:
            for mo in parameters["model"]:
                my_cmd  = "cd " + path + " && "
                my_cmd += "python rnn-smnist.py "
                my_cmd += "--ratio " + str(r) + " "
                my_cmd += "--seed " + str(se) + " "
                my_cmd += "--model " + str(mo) + " "
                for k in parameters.keys():
                    if k.find("ratio") >= 0 or k.find("seed") >= 0 or k.find("model") >= 0:
                        continue
                    my_cmd += "--" + k + " " + str(parameters[k]) + " "
                
                my_cmd += "--gpu " + str(int(available[cnt % len(available)]))
                cnt += 1
                subprocess.Popen(my_cmd, shell=True, stdout=None)


SMNIST_XNN_Parameter = {}
SMNIST_XNN_Parameter["stage1_epoch"] = 150
SMNIST_XNN_Parameter["stage1_lr"]    = 3e-4
SMNIST_XNN_Parameter["dropout"]      = 0.3
SMNIST_XNN_Parameter["nhid"]         = 800
SMNIST_XNN_Parameter["data"]         = "../../data/MNIST"
SMNIST_XNN_Parameter["mode"]         = "test1"    # "train1" for train and "test1" for test
SMNIST_XNN_Parameter["batch_size"]   = 100
SMNIST_XNN_Parameter["clip"]         = 0.25
SMNIST_XNN_Parameter["seed"]         = seed_list
SMNIST_XNN_Parameter["ratio"]        = [0]
SMNIST_XNN_Parameter["model"]        = ["rnn", "snn", "hybrid"] 

# RNN/SNN-Train/Test and HSTNN-Adaptation-Train/Test
#multi_task(path, SMNIST_XNN_Parameter)

ratio_list = [0.05, 0.25, 0.15, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95]

SMNIST_FFS_Parameter = {}
SMNIST_FFS_Parameter["stage1_epoch"] = 150
SMNIST_FFS_Parameter["stage1_lr"]    = 3e-4
SMNIST_FFS_Parameter["dropout"]      = 0.3
SMNIST_FFS_Parameter["nhid"]         = 800
SMNIST_FFS_Parameter["data"]         = "../../data/MNIST"
SMNIST_FFS_Parameter["mode"]         = "test1"#"train1"    # "train1" for train and "test1" for test
SMNIST_FFS_Parameter["batch_size"]   = 100
SMNIST_FFS_Parameter["clip"]         = 0.25
SMNIST_FFS_Parameter["seed"]         = seed_list
SMNIST_FFS_Parameter["ratio"]        = ratio_list
SMNIST_FFS_Parameter["model"]        = ["ffs"]

# FFS-Train/Test
#multi_task(path, SMNIST_FFS_Parameter)

SMNIST_HBR_Parameter = {}
SMNIST_HBR_Parameter["stage2_epoch"] = 150
SMNIST_HBR_Parameter["stage2_lr"]    = 3e-4
SMNIST_HBR_Parameter["dropout"]      = 0.3
SMNIST_HBR_Parameter["nhid"]         = 800
SMNIST_HBR_Parameter["data"]         = "../../data/MNIST"
SMNIST_HBR_Parameter["mode"]         = "test2"#"train2"    # "train2" for train and "test2" for test
SMNIST_HBR_Parameter["batch_size"]   = 100
SMNIST_HBR_Parameter["clip"]         = 0.25
SMNIST_HBR_Parameter["seed"]         = seed_list
SMNIST_HBR_Parameter["ratio"]        = ratio_list
SMNIST_HBR_Parameter["model"]        = ["hybrid"]

# HSTNN-Restoration-Train/Test
multi_task(path, SMNIST_HBR_Parameter)


