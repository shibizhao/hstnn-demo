# Hybrid Spatiotemporal Neural Network (HSTNN)

## Introduction
Hybrid Spatiotemporal Neural Network (HSTNN) enables fine-grained hybridization of RNNs/LSTMs and SNNs. The three-stage training approach offers the flexible trade-off among accuracy, robustness, and computational cost.

## Project Structure
Currently, this repository demonstrates the main results of HSTNN accuracy in four selected datasets. The training approaches and codes are all listed in `experiments`. Each sub-directory in `experiments` are named as:
```
[RNN Type]-[Dataset]-[#Neurons-per-Layer]-[SNN Lens]-[SNN Decay&Threshold]
```
Note that the Len is set to **0.5*k** (Equation 9 of the manuscript), where k is the shape parameter of spiking surrogate functions.
### HSTNN Main File
In each case, the main training and inference python file is `[RNN Type]-[Dataset].py` (e.g. `rnn-ptb.py`). Users can read the implementations of neurons, networks, and the three-stage training process. 
### Parallel Task Manager
In order to enable multiple parallel GPU training or inference tasks, the control file `mprun-[RNN Type]-[Dataset].py` (e.g. `mprun-rnn-smnist.py`) can control the emit of multiple training/inference tasks. The hyperparameters (epochs, learning rate, dropout, clip, etc) are collected in the corresponding dictionaries.
### Result Aggregator
After multiple training and inference tasks, there is an automatic result aggregator for data statistics. Currently, it supports two types (latex table scripts and csv files). The results in csv are stored in `results` directory.


## Project and Environment Setup

Clone the github repo by:
```
git clone https://github.com/shibizhao/hstnn.git
cd hstnn
git submodule update --init --recursive
```

In the author's practice, the virtual environment (e.g. Anaconda) is recommended.
```
conda create -n hstnn python=3.9.16
conda activate hstnn
```
Install the dependencies in requirements.txt:
```
pip install -r requirements.txt
```
Enable the environment variables by:
```
source setup.sh
```

## Dataset Download
We use four datasets for evaluation: Penn-Treebank, Sequential MNIST, Neuromorphic MNIST, and DvsGesture.


### Penn-Treebank
Download the Penn-Treebank dataset:
[Reference Website](https://autonlp.ai/datasets/the-penn-treebank-project)

Move the dataset files to `data/penn-treebank`
```
mkdir -p data/penn-treebank
mv <PTB Download Directory>/ptb.*.txt data/penn-treebank
```

### (Sequential) MNIST
The `torchvision.datasets` will automatically download the MNIST dataset to the `data/MNIST` directory if `args.data` is set as `../../data/MNIST/`.
In order to eliminiate the read/write conflict of the dataset in the multiple parallel tasks, please enter `data/` directory and execute:
```
cd data/
python smnist-dataset-gen.py
``` 

### DVS-Gesture
Download the DVS-Gesture dataset from [IBM Link](https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8) and extract the files in DvsGesture.tar.gz into `./data/DvsGesture`

In the first execution (training or test process), it will check if `./data/DvsGesture/dvs_gestures_events.hdf5` exists and generate a new one if it does not exist.
Similarly, to eliminiate the read/write conflict of the dataset in the multiple parallel tasks, please enter `data/` directory and execute:
```
cd data/
python3 dvs-hdf5-gen.py
```

## How to Run
### Pure RNNs/LSTMs and SNNs
Setup the environment variables and enter the corresponding experiment folder:
```
cd experiments/rnn-ptb-120-0.5-0.6-bias
```
Edit the code of parallel task manager, especially the avaiable gpus, seeds, and training/testing modes.
For example, if there are two avaible GPUs, 
```
gpus = 2
```
To train the RNNs using seed=1111, 2222, 3333, 4444, 5555:
```
PTB_XNN_Parameter["nhid"] = 120
PTB_XNN_Parameter["mode"] = "train1" 
PTB_XNN_Parameter["seed"] = seed_list # or [1111, 2222, 3333, 4444, 5555]
PTB_XNN_Parameter["ratio"] = [0] # DO NOT set a list with more than one element
PTB_XNN_Parameter["model"] = ["rnn"]
```
**It is worth to mention that the "nhid" setting is the number of neurons per layer. Considering the "nlayers" are set as 2 by default, there will be 240 RNNs in this network.**

There will be 5 parallel tasks and they will be dispatched on the two GPUs in a round-robin fashion.


If you want to train the SNNs with the same settings (network architectures, learning rate, training epochs, and etc), please extend the list of "model", like:
```
PTB_XNN_Parameter["model"] = ["rnn", "snn"]
```
To enable these parallel training tasks, please uncomment the execution statement.
```
# RNN/SNN-Train/Test and HSTNN-Stage1-Train/Test
multi_task(path, PTB_XNN_Parameter)
```
Finally, execute the script in the current directory:
```
python mprun-rnn-ptb.py
```
During the training processes, we can get the training log in the folder `log/`, such as: `train_ptb_rnn_1111.log`

After training, we can test the performance of the best model by modifying the working mode as `test1` and re-run the script: `python mprun-rnn-ptb.py`
```
PTB_XNN_Parameter["mode"] = "test1" 
```
And the performance will also be saved in the `log` folder and named as: `test_ptb_rnn_1111.log`

### Directly Hybrid Networks
To train and evaluate the performance of the directly hybrid networks (marked as ffs in the parallel task manager), please set the `ratio` properly, which denotes **SNN ratio** in each layer. For example, 
```
PTB_FFS_Parameter["nhid"] = 120
PTB_FFS_Parameter["mode"] = "train1" 
PTB_FFS_Parameter["seed"] = seed_list # or [1111, 2222, 3333, 4444, 5555]
PTB_FFS_Parameter["ratio"] = [0.75] 
PTB_FFS_Parameter["model"] = ["ffs"]
```
In this setting, the directly hybrid networks will have two RNN/SNN hybrid layers, and there are 90 SNNs and 30 RNNs in each layer. **Please ensure the products of `nhid` and `ratio` are integers.** If you want to training the instances with different SNN ratios, please extend the list of `ratio`. Currently, the authors mainly select `0.05, 0.25, 0.5, 0.75, 0.95` as the basic list and `0.15, 0.35, 0.55, 0.65, 0.85` as the extended list.

Similarly, to enable the parallel tasks, please uncomment the corresponding statement:
```
# FFS-Train/Test
multi_task(path, PTB_FFS_Parameter)
```

After training, we can also get the performance on the testset using the best model by setting the working mode as `test1`. And the training/testing logs for each instances will be saved in the `log/` folder, such as:
`train_ptb_ffs_1111_0.05.log` and `test_ptb_ffs_1111_0.05.log`.

### HSTNNs
Considering there are three stages (Adaption, Selection, and Restoration) in the training process for HSTNNs, the first training stage can be merged into the training of Pure RNN/SNNs (because there is no ratio in the redundant HSTNNs), while the thrid training stage is similar with the training of the directly hybrid networks. Therefore, to get a trained HSTNNs in Adaption stage, we modify or extend the `model` list in `PTB_XNN_Parameter` by:
```
PTB_XNN_Parameter["model"] = ["rnn", "snn", "hybrid"]
```
With the setting of `nhid` and `nlayer`, the architecture of the redundant HSTNNs are constructed: There are `nlayer` hybrid layers, and there are `nhid` RNNs/LSTMs and `nhid` SNNs in each layer.

Currently, the first training stages of RNN/LSTM, SNN, and Redundant HSTNN are all sharing the same hyper-parameters. And similarly, after the training in the adaption stage, we can use `test1` to evaluate the performance of the redundant HSTNN, which will be reported in `log/test_ptb_hybrid_1111.log` (without the ratio).

The selection and the restoration stage are both integrated in the working mode: `train2`. **It is worth to mention that only the HSTNNs have the train2 and test2 modes. And the selection and restoration stages must be enabled after the adapation stage is finished.**
```
PTB_HBR_Parameter["mode"] = "train2" 
```

After the training in the restration stage, we can get the final HSTNN performance by setting the working mode as `test2`.

Similarly, to enable the parallel tasks, please uncomment the corresponding statement:
```
# HBR-Stage2-Train/Test
multi_task(path, PTB_HBR_Parameter)
```

And the training/testing logs for each instances will be saved in the `log/` folder, such as:
`train_ptb_hybrid_1111_0.05.log` and `test_ptb_hybrid_1111_0.05.log`.

### Misc
1. The checkpoints are saved in the `checkpoint/` directory.
2. **Please set the statememts of the `multi_task(xxx)` properly. When we need to training HSTNNs in the restoration stage, please comment the other `multi_task(xxx)` statements.**
3. The `traces/` directory stores the Hessian trace, and used by `pyhessian/`.
4. There will be a script to summarize the performance of pure NNs, directly hybrid networks, and HSTNNs `latex-code-gen.py`. It would be best if all of the tasks of `test1` and `test2` are finished. Just run: `python latex-code-gen.py`.

## Practices and Profiling
Considering the training of HSTNNs with multiple random seeds and hybrid ratios are very time-consuming, here the authors listed some of their practices and profiling to help the users.

**Software** The HSTNN enviroment is set up as a virtual enviroment in Anaconda 22.9.0. The version of Python, PyTorch, and CUDA are 3.9.16, 1.13.1, and 11.7, respectively.

**Hardware** All of the experiments are executed on a server equipped with two-way Intel Xeon Gold 6248R@3.0GHz and 768GB DDR4-2933 Memory. There are two NVIDIA RTX 3080Ti GPUs with 12GB GDDR6 device memory per GPU.

**CPU Core, CUDA Memory, and Main Memory Usage** 
The profiled data is only reference values, especially for the epoch time, which can be easily affected by workload, GPU utilization, temporature, and other factors. The listed epoch time is all collected at the high-workload situation.

| Experiment | CPU Core | CUDA Memory | Main Memory | Epoch Time |
| :----:| :----: | :----: | :----: | :----: |
|RNN-PTB | 1 per instance | 600 MB ~ 650 MB| 1.9 GB ~ 2.0 GB | 30s ~ 60s |
|RNN-SMNIST| 1 per instance | 350 MB ~ 400 MB | 1.9 GB ~ 2.0 GB | 33s ~ 47s |
|RCNN-DVSGes| 1 per instance | 7000 MB ~ 11500 MB | 3.8 GB ~ 4.4 GB | 20s ~ 30s (per 20 epochs)|
|RCNN-NMNIST| 1 per instance | 4000 MB ~ 9000 MB | 11.0 GB ~ 12.0 GB | 100s ~ 200s|
