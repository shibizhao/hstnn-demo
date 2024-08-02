import os
import subprocess

path = "./"

gpus = 4

available = [i for i in range(gpus)]
seed_list = [1111 * i for i in range(1, 7)]

cnt = 0

def multi_task(path, parameters):
    global cnt
    for r in parameters["ratio"]:
        for se in parameters["seed"]:
            for mo in parameters["model"]:
                my_cmd  = "cd " + path + " && "
                my_cmd += "python rcnn-dvs-ges.py "
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


DVS_GESTURE_XNN_Parameter = {}
DVS_GESTURE_XNN_Parameter["stage1_epoch"] = 6400
DVS_GESTURE_XNN_Parameter["stage1_lr"]    = 5e-4
DVS_GESTURE_XNN_Parameter["dropout"]      = 0.2
DVS_GESTURE_XNN_Parameter["mode"]         = "train1"    # "train1" for train and "test1" for test
DVS_GESTURE_XNN_Parameter["batch_size"]   = 36
DVS_GESTURE_XNN_Parameter["seed"]         = seed_list[5:]
DVS_GESTURE_XNN_Parameter["ratio"]        = [0]
DVS_GESTURE_XNN_Parameter["model"]        = ["hybrid"] 

# RNN/SNN-Train/Test and HSTNN-Adaptation-Train/Test
multi_task(path, DVS_GESTURE_XNN_Parameter)

ratio_list = [0.0625, 0.125, 0.25, 0.375, 0.50, 0.625, 0.875, 0.9375]

DVS_GESTURE_FFS_Parameter = {}
DVS_GESTURE_FFS_Parameter["stage1_epoch"] = 6400
DVS_GESTURE_FFS_Parameter["stage1_lr"]    = 5e-4
DVS_GESTURE_FFS_Parameter["dropout"]      = 0.2
DVS_GESTURE_FFS_Parameter["mode"]         = "test1"    # "train1" for train and "test1" for test
DVS_GESTURE_FFS_Parameter["batch_size"]   = 36
DVS_GESTURE_FFS_Parameter["seed"]         = seed_list
DVS_GESTURE_FFS_Parameter["ratio"]        = ratio_list
DVS_GESTURE_FFS_Parameter["model"]        = ["ffs"]

# FFS-Train/Test
# multi_task(path, DVS_GESTURE_FFS_Parameter)

DVS_GESTURE_HBR_Parameter = {}
DVS_GESTURE_HBR_Parameter["stage2_epoch"] = 6400
DVS_GESTURE_HBR_Parameter["stage2_lr"]    = 3e-4
DVS_GESTURE_HBR_Parameter["dropout"]      = 0.2
DVS_GESTURE_HBR_Parameter["mode"]         = "test2"
DVS_GESTURE_HBR_Parameter["batch_size"]   = 36
DVS_GESTURE_HBR_Parameter["seed"]         = seed_list
DVS_GESTURE_HBR_Parameter["ratio"]        = ratio_list
DVS_GESTURE_HBR_Parameter["model"]        = ["hybrid"]

# HSTNN-Restoration-Train/Test
# multi_task(path, DVS_GESTURE_HBR_Parameter)
