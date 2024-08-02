# coding: utf-8
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
import torch.nn.functional as F
import logging
from torch.utils.data.dataloader import default_collate

torch.multiprocessing.set_start_method('spawn')
torch.set_num_threads(1)
###############################################################################
# Define Activation Function for SNNs
###############################################################################
lens = 0.25
class ActFun(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return input.gt(0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        temp = abs(input) < lens
        return grad_input * temp.float() / (2 * lens)

WinFunc = ActFun.apply

###############################################################################
# Define HNN Model
###############################################################################

class HybridLayer(nn.Module):
    # hybrid RCNN and SCNN block
    def __init__(self, input_channels, rcnn_channels, scnn_channels, conv_kernel_size, pool_kernel_size, conv_stride, pool_stride, fc_padding, fv_padding, avr_pool=True, device="cuda"):
        super(HybridLayer, self).__init__()
        self.rcnn_channels = rcnn_channels
        self.scnn_channels = scnn_channels
        self.conv_kernel_size = conv_kernel_size
        self.pool_kernel_size = pool_kernel_size
        self.conv_stride = conv_stride
        self.pool_stride = pool_stride
        self.avr_pool = avr_pool
        self.fc_padding = fc_padding
        self.fv_padding = fv_padding
        self.device = device

        self.act_fun = WinFunc
        
        self.decay = 0.3
        self.thresh = 0.3
        self.lens   = 0.25

        if rcnn_channels > 0:
            self.rcnn_fc = nn.Conv2d(input_channels, rcnn_channels, kernel_size=conv_kernel_size, stride=conv_stride, padding=fc_padding, device=device)
            self.rcnn_fv = nn.Conv2d(rcnn_channels,  rcnn_channels, kernel_size=conv_kernel_size, stride=conv_stride, padding=fv_padding, device=device)
        if scnn_channels > 0:
            self.scnn_fc = nn.Conv2d(input_channels, scnn_channels, kernel_size=conv_kernel_size, stride=conv_stride, padding=fc_padding, device=device)
        if avr_pool == True:
            self.scnn_avg_pool = nn.AvgPool2d(kernel_size=pool_kernel_size, stride=pool_stride)
            self.rcnn_avg_pool = nn.AvgPool2d(kernel_size=pool_kernel_size, stride=pool_stride)

    def scnn_mem_update(self, conv, x, mem, spike):
        mem = mem * self.decay * (1 - spike) + conv(x)
        spike = self.act_fun(mem - self.thresh)
        return mem, spike

    def rcnn_update(self, conv1, conv2, x, y):
        return torch.relu(conv1(x) + conv2(y))

    def forward(self, inputs, scnn_mem, scnn_spike, rcnn_y):
        # SCNN Processing inputs
        if self.scnn_channels > 0:
            scnn_mem, scnn_spike = self.scnn_mem_update(self.scnn_fc, inputs, scnn_mem, scnn_spike)
            if self.avr_pool == True:
                pooled_scnn_spike = self.scnn_avg_pool(scnn_spike)
            else:
                pooled_scnn_spike = scnn_spike
        else:
            pooled_scnn_spike = torch.zeros([scnn_spike.shape[0], scnn_spike.shape[1], scnn_spike.shape[2] // self.pool_kernel_size, scnn_spike.shape[3] // self.pool_kernel_size], device=self.device)
        if self.rcnn_channels > 0:
            rcnn_y               = self.rcnn_update(self.rcnn_fc, self.rcnn_fv, inputs, rcnn_y)
            if self.avr_pool == True:
                pooled_rcnn_y    = self.rcnn_avg_pool(rcnn_y)
            else:
                pooled_rcnn_y = rcnn_y
        else:
            pooled_rcnn_y = torch.zeros([rcnn_y.shape[0], rcnn_y.shape[1], rcnn_y.shape[2] // self.pool_kernel_size, rcnn_y.shape[3] // self.pool_kernel_size], device=self.device)
    
        return scnn_mem, scnn_spike, rcnn_y, pooled_scnn_spike, pooled_rcnn_y


class HNNModel(nn.Module):
    def __init__(self, nclass, nwin, layer_config, linear_config, buffer_config, dropout=None, device="cuda", union=False):
        super(HNNModel, self).__init__()

        self.nclass    = nclass # 10
        self.nwin      = nwin   # 15

        self.dropout   = nn.Dropout(dropout)
        self.device    = torch.device(device)
        
        self.hcnn_layers = nn.ModuleList([HybridLayer(input_channels   = layer['input_channels'],
                                        rcnn_channels    = layer['rcnn_channels'],
                                        scnn_channels    = layer['scnn_channels'],
                                        conv_kernel_size = layer['conv_kernel_size'],
                                        pool_kernel_size = layer['pool_kernel_size'],
                                        conv_stride      = layer['conv_stride'],
                                        pool_stride      = layer['pool_stride'],
                                        fc_padding       = layer['fc_padding'], 
                                        fv_padding       = layer['fv_padding'], 
                                        avr_pool         = layer['avr_pool'],
                                        device           = device)
                                        for layer in layer_config])     

        self.linear_layers = nn.ModuleList([nn.Linear(linear['input_channels'], 
                                                      linear['output_channels'], 
                                                      device=device)
                                            for linear in linear_config])
    
        self.layer_config = layer_config

        self.buffer_config = buffer_config

        self.linear_config = linear_config

    def forward(self, inputs):
        batch_size, _, _, _, _ = inputs.shape
        scnn_mem_list   = [torch.zeros([batch_size, 
                                      buffer['scnn_channels'], 
                                      buffer['output_size'], 
                                      buffer['output_size']], device=self.device) 
                           for buffer in self.buffer_config]
        scnn_spike_list = [torch.zeros([batch_size, 
                                      buffer['scnn_channels'], 
                                      buffer['output_size'], 
                                      buffer['output_size']], device=self.device) 
                           for buffer in self.buffer_config]
        rcnn_y_list     = [torch.zeros([batch_size, 
                                      buffer['rcnn_channels'], 
                                      buffer['output_size'], 
                                      buffer['output_size']], device=self.device) 
                           for buffer in self.buffer_config]

        scnn_accum      = torch.zeros([batch_size, 
                                      self.buffer_config[-1]['scnn_channels'], 
                                      self.buffer_config[-1]['pooled_size'], 
                                      self.buffer_config[-1]['pooled_size']], device=self.device)

        for step in range(self.nwin):
            x = inputs[:,:,:,:,step]
            for id in range(len(self.hcnn_layers)):
                scnn_mem_list[id], scnn_spike_list[id], rcnn_y_list[id], pooled_scnn_spike, pooled_rcnn_y = \
                    self.hcnn_layers[id](x,
                                         scnn_mem_list[id],
                                         scnn_spike_list[id], 
                                         rcnn_y_list[id])
                x = torch.cat((pooled_scnn_spike, pooled_rcnn_y), dim=1)
            scnn_accum += pooled_scnn_spike
        
        x = torch.cat((scnn_accum, pooled_rcnn_y), dim=1).view(batch_size, -1)
        x = F.relu(self.linear_layers[0](x))
        x = self.linear_layers[1](x)
        return x
    def report_neuron_numbers(self):
        rcnn_number = 0
        scnn_number = 0
        for layer in self.layer_config:
            rcnn_number += layer["rcnn_channels"]
            scnn_number += layer["scnn_channels"]
        return {"rcnn": rcnn_number, "scnn": scnn_number}

def print_hcnn_weight_name(layer_id, neuron_type, connect_type):
    return "hcnn_layers." + str(layer_id) + "." + str(neuron_type) + "_" +  str(connect_type)



###############################################################################
# Parse arguments
###############################################################################

parser = argparse.ArgumentParser(description='HNN Model on N-MNIST Dataset')

parser.add_argument('--data', type=str,
                    default='../../data/N-MNIST/processed/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='hybrid', help='type of network (rnn, snn, ffs[fixed-from-scratch], hybrid)')

parser.add_argument('--mode', type=str, default='train1', help='type of operation (train1, test1, train2, test2)')
# Activate parameters
parser.add_argument('--batch_size', type=int, default=100, metavar='N', help='batch size')
parser.add_argument('--stage1_epochs', type=int, default=100, help='training epochs of Adaptation stage')
parser.add_argument('--stage2_epochs', type=int, default=100, help='training epochs of Restoration stage')
parser.add_argument('--stage1_lr', type=float, default=8e-4, help='learning rate of Adaptation stage')
parser.add_argument('--stage2_lr', type=float, default=8e-4, help='learning rate of Restoration stage')

parser.add_argument('--ratio', type=float, default=0.5, help='snn ratio of hybird network')

# Default parameters
parser.add_argument('--nwin', type=int, default=15,  help='value of time window')
parser.add_argument('--dt',   type=int, default=3,   help='value of dt (2ms, 3ms)')

parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.2, help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=1111, help='random seed')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--log-interval', type=int, default=100, metavar='N', help='report interval')
parser.add_argument('--nlayers', type=int, default=2, help='number of layers')
parser.add_argument('--nclass', type=int, default=10, help="MNIST class, 10, NOT MODIFY")
parser.add_argument('--gpu', type=str, default='0', help='gpu number')

args = parser.parse_args()
assert args.nclass == 10

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

setup_seed(args.seed)
device = torch.device("cuda")


###############################################################################
# Load data
###############################################################################

def data_prepare(arg_args):
    from MyLargeDataset import MyDataset

    data_path = arg_args.data + "/" + str(args.dt) + "ms" + "/"

    train_dataset = MyDataset(data_path, "nmnist_h", args.nwin)
    test_dataset  = MyDataset(data_path, "nmnist_r", args.nwin)
    
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=arg_args.batch_size,
                                            shuffle=True, drop_last=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=arg_args.batch_size,
                                            shuffle=False, drop_last=True)
    return train_loader, test_loader

trainloader, testloader = data_prepare(args)

criterion = nn.MSELoss()
# criterion = nn.CrossEntropyLoss()

def train(arg_model, optimizers, epoch, logger):
    arg_model.cuda()
    arg_model.train()
    train_loss = 0
    correct    = 0
    total      = 0
    start_time = time.time()

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        arg_model.zero_grad()
        outputs = arg_model(inputs.float().cuda())
        target_loss = criterion(outputs.cpu(), targets)

        both_loss   = target_loss.to(device)
        both_loss.backward()
        torch.nn.utils.clip_grad_norm_(arg_model.parameters(), args.clip)

        optimizers.step()
        optimizers.zero_grad()
        
        train_loss  += target_loss.item()
        _, predicted = outputs.cpu().max(1)
        _, labels    = targets.max(1)
        total       += targets.size(0)

        correct     += (predicted == labels).sum()

        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            cur_loss = train_loss / args.log_interval
            elapsed  = time.time() - start_time
            logger.info('| epoch {:3d} | {:5d}/{:5d} batches | {:5.2f} ms/batch '
                        '| loss {:.5f} | acc {:.3f}%'.format(
                            epoch, batch_idx, int(60000 / args.batch_size), elapsed * 1000 / args.log_interval, cur_loss, 100.0 * correct / total)
                        )
            train_loss = 0
            total      = 0
            correct    = 0
            start_time = time.time()

def evaluate(arg_model):
    # Turn on evaluation mode which disables dropout.
    arg_model.eval()
    test_loss = 0
    correct   = 0
    total     = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            outputs = arg_model(inputs.float().cuda())
            loss        = criterion(outputs.cpu(), targets)

            test_loss  += loss.item()
            _, predicted = outputs.cpu().max(1)
            _, labels    = targets.max(1)
            total       += targets.size(0)

            correct     += (predicted == labels).sum()

        test_loss = test_loss / (batch_idx+1)

    acc = 100.0 * correct / total
    return acc, test_loss

def train_over_epoch(arg_model, epoch_num, optimizer, logger, arg_model_name):
    # record variables
    test_acc_record = []
    best_test_acc = None
    best_model = None
    # train epoch_num epochs
    for epoch in range(1, epoch_num + 1):
        epoch_start_time = time.time()
        train(arg_model, optimizer, epoch, logger)
        test_acc, test_loss = evaluate(arg_model)
        test_acc_record.append(test_acc)
        logger.info('-' * 83)
        logger.info('| end of epoch {:3d} | time: {:5.2f}s | test loss {:.5f} | test acc {:.3f}%'
                     .format(epoch, time.time() - epoch_start_time, test_loss, test_acc))
        logger.info('-' * 83)

        if not best_test_acc or best_test_acc < test_acc:
            best_test_acc = test_acc
            best_model = copy.deepcopy(arg_model)
            state = {
                'net' : arg_model.state_dict(),
                'seed': args.seed,
                'layer_config' : arg_model.layer_config, 
                'buffer_config' : arg_model.buffer_config,
                'linear_config'  : arg_model.linear_config,
                'test_acc_record' : test_acc_record
            }
            if not os.path.isdir('checkpoint'):
                os.system('mkdir -p checkpoint')
            torch.save(state, './checkpoint/' + arg_model_name + ".t7")
        epoch_start_time = time.time()
    test_trained_model(best_model, logger)
    return best_model

def test_trained_model(arg_model, logger):
    test_acc, test_loss = evaluate(arg_model)
    logger.info('=' * 83)
    logger.info('| Performance on Test Set | test loss {:.5f} | test acc {:.5f}'.format(
        test_loss, test_acc))
    logger.info('=' * 83)

def initialize_config():
    import model_config
    import copy
    layer_config  = copy.deepcopy(model_config.base_config)
    buffer_config = copy.deepcopy(model_config.buffer_config)
    linear_config = copy.deepcopy(model_config.linear_config)
    if args.model == "rnn":
        for i in range(len(layer_config)):
            layer_config[i]['rcnn_channels'] = layer_config[i]['cnn_channels']
            layer_config[i]['scnn_channels'] = 0
        for i in range(len(buffer_config)):
            buffer_config[i]['rcnn_channels'] = layer_config[i]['rcnn_channels']
            buffer_config[i]['scnn_channels'] = 0
    elif args.model == "snn":
        for i in range(len(layer_config)):
            layer_config[i]['scnn_channels'] = layer_config[i]['cnn_channels']
            layer_config[i]['rcnn_channels'] = 0
        for i in range(len(buffer_config)):
            buffer_config[i]['scnn_channels'] = layer_config[i]['scnn_channels']
            buffer_config[i]['rcnn_channels'] = 0
    elif args.model == "hybrid":
        for i in range(len(layer_config)):
            layer_config[i]['rcnn_channels'] = layer_config[i]['cnn_channels']
            layer_config[i]['scnn_channels'] = layer_config[i]['cnn_channels']
            if i > 0:
                layer_config[i]['input_channels'] = layer_config[i-1]['rcnn_channels'] + layer_config[i-1]['scnn_channels']
        for i in range(len(buffer_config)):
            buffer_config[i]['scnn_channels'] = layer_config[i]['scnn_channels']
            buffer_config[i]['rcnn_channels'] = layer_config[i]['rcnn_channels']
                  
        linear_config[0]["input_channels"] = (layer_config[-1]['rcnn_channels'] + layer_config[-1]['scnn_channels']) * 4 * 4
    elif args.model == "ffs":
        for i in range(len(layer_config)):
            layer_config[i]['rcnn_channels'] = int(layer_config[i]['cnn_channels'] * (1 - args.ratio))
            layer_config[i]['scnn_channels'] = int(layer_config[i]['cnn_channels'] * args.ratio)
        for i in range(len(buffer_config)):
            buffer_config[i]['rcnn_channels'] = layer_config[i]['rcnn_channels']
            buffer_config[i]['scnn_channels'] = layer_config[i]['scnn_channels']
    return layer_config, buffer_config, linear_config

      
def train_origin_model(model_name, logger):
    layer_config, buffer_config, linear_config = initialize_config()

    origin_model = HNNModel(nclass=args.nclass, nwin=args.nwin, layer_config=layer_config, linear_config=linear_config, buffer_config=buffer_config, dropout=args.dropout, device="cuda").to(device)

    print(origin_model)

    optimizer    = optim.Adam(origin_model.parameters(), lr=args.stage1_lr)
    #optimizer    = optim.SGD(origin_model.parameters(), lr=args.stage1_lr, momentum=0.9)

    trained_model = train_over_epoch(origin_model, args.stage1_epochs, optimizer, logger, model_name)
    return trained_model

def train_pruned_model(arg_model, model_name, logger):
    optimizer = optim.Adam(arg_model.parameters(), lr=args.stage2_lr)
    trained_model = train_over_epoch(arg_model, args.stage2_epochs, optimizer, logger, model_name)
    return trained_model

def print_eff_index(name, l, logger):
    st = "["
    for i in l:
        st += str(i) + ", "
    st = st[:-2] + "]"
    logger.info(name + ": " + st)


def compute_effective_indices(arg_model, logger, hessian_mode="trace", neuron_nums_dict=None):
    assert args.model == "hybrid"
    from hessian_pruner import HessianPruner
    if not os.path.isdir('traces'):
        os.system('mkdir -p traces')
    # todo: func!!!
    trace_file_name = "./traces/" + "trace" + "_" + "hybrid" + "_"  + str(args.seed) + ".npy"
    pruner = HessianPruner(arg_model, trace_file_name, hessian_mode=hessian_mode, neuron_nums_dict=neuron_nums_dict)
    eff_dict = pruner.make_pruned_model(trainloader, criterion, device, args.ratio, args.seed)

    for k, v in eff_dict.items():
        print_eff_index(k, sorted(v), logger)
    
    for k, v in eff_dict.items():
        logger.info(str(k) + str(len(v)))

    return eff_dict



def logger_generation(file_name):
    if not os.path.isdir('log'):
        os.system('mkdir -p log')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    sh = logging.StreamHandler()
    sh.setLevel(logging.DEBUG)

    fh = logging.FileHandler("./log/" + file_name + ".log")
    fh.setLevel(logging.DEBUG)
    
    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger

def load_model(arg_model_name, make_union=False):
    ckpt_path = "./checkpoint/" + arg_model_name + ".t7"
    ckpts     = torch.load(ckpt_path, map_location=torch.device('cpu'))
    weights_dict  = ckpts["net"]
    layer_config  = ckpts["layer_config"]
    buffer_config = ckpts["buffer_config"]
    linear_config = ckpts["linear_config"]
    tmp_model     = HNNModel(nclass=args.nclass, nwin=args.nwin, layer_config=layer_config, linear_config=linear_config, buffer_config=buffer_config, dropout=args.dropout, device="cuda").to(device)
    if make_union == False:
        tmp_model.load_state_dict(weights_dict)
    else:
        tmp_model.load_union_state(weights_dict)
    return tmp_model 


def load_and_shrink(arg_model_name, arg_eff_dict):
    ckpt_path = "./checkpoint/" + arg_model_name + ".t7"
    ckpts     = torch.load(ckpt_path, map_location=torch.device('cpu'))

    new_state_dict = copy.deepcopy(ckpts['net'])
    layer_config  = ckpts["layer_config"]
    buffer_config = ckpts["buffer_config"]
    linear_config = ckpts["linear_config"]


    assert len(layer_config) == len(buffer_config)

    for layer_idx in range(len(layer_config)):
        
        # save the original one
        layer_config[layer_idx]["old_rcnn_channels"] = layer_config[layer_idx]["rcnn_channels"]
        layer_config[layer_idx]["old_scnn_channels"] = layer_config[layer_idx]["scnn_channels"]
        
        # load the pruned one
        rcnn_channels = len(arg_eff_dict[print_hcnn_weight_name(layer_idx, "rcnn", "fc")])
        scnn_channels = len(arg_eff_dict[print_hcnn_weight_name(layer_idx, "scnn", "fc")])
        
        layer_config[layer_idx]["rcnn_channels"] = rcnn_channels
        layer_config[layer_idx]["scnn_channels"] = scnn_channels
        
        buffer_config[layer_idx]["rcnn_channels"] = rcnn_channels
        buffer_config[layer_idx]["scnn_channels"] = scnn_channels
        
        layer_config[layer_idx]["old_input_channels"] = layer_config[layer_idx]["input_channels"]
        if layer_idx > 0:
            # only update the second and the later layers
            layer_config[layer_idx]["input_channels"] = layer_config[layer_idx - 1]["rcnn_channels"] \
                                                      + layer_config[layer_idx - 1]["scnn_channels"]
    
    linear_config[0]["old_input_channels"] = linear_config[0]["input_channels"]
    linear_config[0]["input_channels"] = (layer_config[-1]['rcnn_channels'] + layer_config[-1]['scnn_channels']) * 4 * 4


    model = HNNModel(nclass=args.nclass, nwin=args.nwin, layer_config=layer_config, linear_config=linear_config, buffer_config=buffer_config, dropout=args.dropout, device="cuda")



    for layer_idx in range(len(layer_config)):
        # conv weight order: output_channels, input_channels, filter_size, filter_size
        rcnn_output_mask = torch.zeros([layer_config[layer_idx]["old_rcnn_channels"],  layer_config[layer_idx]["rcnn_channels"]])
        scnn_output_mask = torch.zeros([layer_config[layer_idx]["old_scnn_channels"],  layer_config[layer_idx]["scnn_channels"]])
        rcnn_index_list = arg_eff_dict[print_hcnn_weight_name(layer_idx, "rcnn", "fc")]
        scnn_index_list = arg_eff_dict[print_hcnn_weight_name(layer_idx, "scnn", "fc")]
        for idx in range(len(rcnn_index_list)):
            rcnn_output_mask[rcnn_index_list[idx]][idx] = 1
        for idx in range(len(scnn_index_list)):
            scnn_output_mask[scnn_index_list[idx]][idx] = 1
        
        if layer_idx > 0:
            comm_input_mask  = torch.zeros([layer_config[layer_idx]["old_input_channels"], layer_config[layer_idx]["input_channels"]])
            prev_rcnn_index_list = arg_eff_dict[print_hcnn_weight_name(layer_idx - 1, "rcnn", "fc")]
            prev_scnn_index_list = arg_eff_dict[print_hcnn_weight_name(layer_idx - 1, "scnn", "fc")]
            prev_offset          = layer_config[layer_idx-1]["old_scnn_channels"]
            
            input_index_list = prev_scnn_index_list + [i + prev_offset for i in prev_rcnn_index_list]
            for idx in range(len(input_index_list)):
                comm_input_mask[input_index_list[idx]][idx] = 1
        else:
            assert layer_config[layer_idx]["old_input_channels"] == layer_config[layer_idx]["input_channels"]
            comm_input_mask = torch.eye(layer_config[layer_idx]["old_input_channels"])

        def process_hcnn_weights(arg_ckpts, arg_layer_idx, arg_type, arg_input_mask, arg_output_mask, arg_ckpt_dicts):
            # fc_w => [out_ch, in_ch, f_size_row, f_size_col] -> [f_size_col, f_size_row, in_ch, out_ch]
            fc_w = arg_ckpts['net'][print_hcnn_weight_name(arg_layer_idx, arg_type, "fc") + ".weight"].permute(3, 2, 1, 0)
            # fc_w -> [f_size_col, f_size_row, in_ch, pruned_out_ch] -> [f_size_col, f_size_row, pruned_out_ch, in_ch]
            fc_w = torch.matmul(fc_w, arg_output_mask).permute(0, 1, 3, 2)
            # rcnn_fc -> [f_size_col, f_size_row, pruned_out_ch, pruned_in_ch] -> [pruned_out_ch, pruned_in_ch, f_size_row, f_size_col]
            fc_w = torch.matmul(fc_w, arg_input_mask).permute(2, 3, 1, 0)
            
            # fc_b => [out_ch] -> [1, out_ch]
            fc_b = arg_ckpts['net'][print_hcnn_weight_name(arg_layer_idx, arg_type, "fc") + ".bias"].unsqueeze(0)
            # fc_b -> [1, out_ch] -> [1, pruned_out_ch] -> 
            fc_b = torch.matmul(fc_b, arg_output_mask).squeeze(0)

            arg_ckpt_dicts[print_hcnn_weight_name(arg_layer_idx, arg_type, "fc") + ".weight"] = fc_w
            arg_ckpt_dicts[print_hcnn_weight_name(arg_layer_idx, arg_type, "fc") + ".bias"] = fc_b

            if arg_type == "rcnn":
                fv_w = arg_ckpts['net'][print_hcnn_weight_name(arg_layer_idx, arg_type, "fv") + ".weight"].permute(3, 2, 1, 0)
                fv_w = torch.matmul(fv_w, arg_output_mask).permute(0, 1, 3, 2)
                fv_w = torch.matmul(fv_w, arg_output_mask).permute(2, 3, 1, 0)
                fv_b = arg_ckpts['net'][print_hcnn_weight_name(arg_layer_idx, arg_type, "fv") + ".bias"].unsqueeze(0)
                fv_b = torch.matmul(fv_b, arg_output_mask).squeeze(0)
                arg_ckpt_dicts[print_hcnn_weight_name(arg_layer_idx, arg_type, "fv") + ".weight"] = fv_w
                arg_ckpt_dicts[print_hcnn_weight_name(arg_layer_idx, arg_type, "fv") + ".bias"] = fv_b
                return fc_w, fc_b, fv_w, fv_b
            else:
                return fc_w, fc_b, None, None
            
    
        process_hcnn_weights(ckpts, layer_idx, "rcnn", comm_input_mask, rcnn_output_mask, new_state_dict)
        process_hcnn_weights(ckpts, layer_idx, "scnn", comm_input_mask, scnn_output_mask, new_state_dict)

    # prune the first linear layer
    comm_input_mask  = torch.zeros([linear_config[0]["old_input_channels"] // 16, linear_config[0]["input_channels"] // 16])
    prev_rcnn_index_list = arg_eff_dict[print_hcnn_weight_name(len(layer_config) - 1, "rcnn", "fc")]
    prev_scnn_index_list = arg_eff_dict[print_hcnn_weight_name(len(layer_config) - 1, "scnn", "fc")]
    prev_offset          = layer_config[-1]["old_scnn_channels"]
    
    input_index_list = prev_scnn_index_list + [i + prev_offset for i in prev_rcnn_index_list]
    for idx in range(len(input_index_list)):
        comm_input_mask[input_index_list[idx]][idx] = 1
    
    # l0_w => [output, flatten_inputs] -> [output, input, 5_row, 5_col] -> [5_col, 5_row, output, input]
    l0_w = ckpts['net']["linear_layers.0.weight"].view(linear_config[0]["output_channels"], linear_config[0]["old_input_channels"] // 16, 4, 4).permute(3, 2, 0, 1)
    # l0_w => [5_col, 5_row, output, input] -> [5_col, 5_row, output, pruned_input] -> [output, pruned_input, 5_row, 5_col] -> [output, -1]
    l0_w = torch.matmul(l0_w, comm_input_mask).permute(2, 3, 1, 0).reshape(linear_config[0]["output_channels"], -1)

    # no bias pruning for the first linear layer

    new_state_dict["linear_layers.0.weight"] = l0_w

    model.load_state_dict(new_state_dict)

    return model


def get_pruned_model(model_name, logger, union=False, hessian_mode="trace"):
    union_model = load_model(model_name, make_union=union)
    # test_trained_model(union_model, logger)

    neuron_numbers = union_model.report_neuron_numbers()

    neuron_numbers["rcnn"] = int(neuron_numbers["rcnn"] * (1 - args.ratio))
    neuron_numbers["scnn"] = int(neuron_numbers["scnn"] * (args.ratio))

    eff_dict    = compute_effective_indices(union_model, logger, hessian_mode=hessian_mode, neuron_nums_dict=neuron_numbers)
    del union_model
    torch.cuda.empty_cache()
    pruned_model = load_and_shrink(model_name, eff_dict)
    test_trained_model(pruned_model, logger)
    return pruned_model

assert args.model in ["rnn", "snn", "ffs", "hybrid"]

dataset_name = "nmnist"


if args.mode == "train1":
    model_name = dataset_name + "_" + args.model + "_" + str(args.seed)
    if args.model == "ffs":
        model_name += "_" + str(args.ratio)
    logfile_name = "train" + "_" + model_name

    train1_logger = logger_generation(logfile_name)
    trained_model = train_origin_model(model_name, train1_logger)

elif args.mode == "test1":
    model_name = dataset_name + "_" + args.model + "_" + str(args.seed)
    if args.model == "ffs":
        model_name += "_" + str(args.ratio)
    logfile_name = "test" + "_" + model_name

    test1_logger  = logger_generation(logfile_name)
    trained_model = load_model(model_name, make_union=False)
    print(trained_model)
    test_trained_model(trained_model, test1_logger)

elif args.mode == "train2":
    assert args.model == "hybrid"
    if args.model == "hybrid":
        input_model_name  = dataset_name + "_" + args.model + "_" + str(args.seed)
        pruned_model_name = dataset_name + "_" + args.model + "_" + str(args.seed) + "_" + str(args.ratio)
        logfile_name = "train" + "_" + pruned_model_name
        train2_logger = logger_generation(logfile_name)
        pruned_model = get_pruned_model(input_model_name, train2_logger, union=False) # caution!
        trained_model = train_pruned_model(pruned_model, pruned_model_name, train2_logger)
    else:
        print("Unknown Mode")
        assert False

elif args.mode == "test2":
    assert args.model == "hybrid"
    if args.model == "hybrid":
        model_name = dataset_name + "_" + args.model + "_" + str(args.seed) + "_" + str(args.ratio)
        logfile_name = "test" + "_" + model_name
        test2_logger = logger_generation(logfile_name)
        final_model  = load_model(model_name, make_union=False)
        test_trained_model(final_model, test2_logger)
    else:
        print("Unknown Mode")
        assert False
else:
    print("Unknown Mode")
    assert False
