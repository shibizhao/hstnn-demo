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
import logging

torch.multiprocessing.set_start_method('spawn')
torch.set_num_threads(8)

###############################################################################
# Define Activation Function for SNNs
###############################################################################
lens = 0.5
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

class HNNModel(nn.Module):

    def __init__(self, nclass, ninp, rnn_shape, snn_shape, dropout=None, device="cuda", func="window", union=False):
        super(HNNModel, self).__init__()

        assert len(rnn_shape) == 2 and len(snn_shape) == 2
        assert rnn_shape[0] + snn_shape[0] > 0 and rnn_shape[1] + snn_shape[1] > 0

        self.nclass    = nclass #10
        self.ninp      = ninp   #28
        self.rnn_shape = rnn_shape
        self.snn_shape = snn_shape
        self.func      = func
        self.dropout   = nn.Dropout(dropout)
        self.device    = torch.device(device)
        self.union     = union

        self.nlayers   = 2
        self.decay     = 0.6
        self.thresh    = 0.6

        if self.func == "window":
            self.act_fun = WinFunc

        if self.union is False:
            if self.snn_shape[0] > 0:
                self.snn_fc1 = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(ninp, snn_shape[0])))
            if self.snn_shape[1] > 0:
                self.snn_fc2 = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(snn_shape[0] + rnn_shape[0], snn_shape[1])))
            if self.rnn_shape[0] > 0:
                self.rnn_fc1 = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(ninp, rnn_shape[0])))
            if self.rnn_shape[1] > 0:
                self.rnn_fc2 = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(snn_shape[0] + rnn_shape[0], rnn_shape[1]))) # caution!
            if self.rnn_shape[0] > 0:    
                self.rnn_fv1 = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(rnn_shape[0], rnn_shape[0])))
            if self.rnn_shape[1] > 0:
                self.rnn_fv2 = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(rnn_shape[1], rnn_shape[1])))
        else:
            if self.snn_shape[0] > 0:
                self.snn1 = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(ninp, snn_shape[0])))
            if self.snn_shape[1] > 0:
                self.snn2 = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(snn_shape[0] + rnn_shape[0], snn_shape[1])))
            if self.rnn_shape[0] > 0:
                self.rnn1 = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(ninp + rnn_shape[0], rnn_shape[0])))
            if self.rnn_shape[1] > 0:
                self.rnn2 = nn.Parameter(nn.init.kaiming_uniform_(torch.Tensor(snn_shape[0] + rnn_shape[0] + rnn_shape[1], rnn_shape[1]))) # caution!
        self.all_fc = nn.Linear(snn_shape[1] + rnn_shape[1], nclass) # caution!
        
        self.init_weights()
    
    def init_weights(self):
        # initialize the weights
        initrange = 0.1
        self.all_fc.bias.data.zero_()
        self.all_fc.weight.data.uniform_(-initrange, initrange)

    def init_hidden(self, bsz):
        # initialize the hidden tensors to zero
        hidden1 = torch.zeros([bsz, self.rnn_shape[0]], dtype=torch.float32, device=self.device) 
        hidden2 = torch.zeros([bsz, self.rnn_shape[1]], dtype=torch.float32, device=self.device)      
        return (hidden1, hidden2)

    def load_union_state(self, arg_dict):
        rnn_type = "rnn"
        for k in arg_dict.keys():
            if k.find("lstm") >= 0:
                rnn_type = "lstm"
        if rnn_type == "rnn":    
            rnn1 = torch.cat((arg_dict["rnn_fc1"], arg_dict["rnn_fv1"]), dim=0)
            rnn2 = torch.cat((arg_dict["rnn_fc2"], arg_dict["rnn_fv2"]), dim=0)
        else:
            rnn1 = torch.cat((arg_dict["lstm_wi1"], arg_dict["lstm_wh1"]), dim=1)
            rnn2 = torch.cat((arg_dict["lstm_wi2"], arg_dict["lstm_wh2"]), dim=1)
        arg_dict["snn1"] = copy.deepcopy(arg_dict["snn_fc1"])
        arg_dict["snn2"] = copy.deepcopy(arg_dict["snn_fc2"])                        
        union_dict = {k: v for k, v in arg_dict.items() if k in self.state_dict().keys()}
        union_dict[rnn_type + "1"] = rnn1
        union_dict[rnn_type + "2"] = rnn2
        self.load_state_dict(union_dict)

    def rnn_update(self, fc, fv, inputs, last_state):
        state = inputs.mm(fc) + last_state.mm(fv)
        activation = state.sigmoid()
        return activation

    def rnn_union_update(self, uf, inputs, last_state):
        ui    = torch.cat((inputs, last_state), dim=1)
        state = ui.mm(uf)
        activation = state.sigmoid()
        return activation


    def snn_update(self, fc, inputs, mem, spike):
        state = inputs.mm(fc)
        mem = mem * (1 - spike) * self.decay + state
        now_spike = self.act_fun(mem - self.thresh)
        return mem, now_spike.float()

    def forward(self, input, hidden):

        buf_input = input.view(-1, 28, 28)
        buf_input = buf_input.transpose(1,2)
        time_window = 28

        batch_size, col_size, row_size = buf_input.shape

        h1_mem = h1_spike = torch.zeros(batch_size, self.snn_shape[0], device=device)
        h2_mem = h2_spike = torch.zeros(batch_size, self.snn_shape[1], device=device)
        h2_sum_spike      = torch.zeros(batch_size, self.snn_shape[1], device=device) # caution!
        h1_y              = torch.zeros(batch_size, self.rnn_shape[0], device=device)
        h2_y              = torch.zeros(batch_size, self.rnn_shape[1], device=device)
        
        if self.union is False:
            for step in range(time_window):
                output0 = buf_input[:,  step,:]
                output0 = output0.view(batch_size, -1)
                if self.snn_shape[0] > 0:
                    h1_mem, h1_spike = self.snn_update(self.snn_fc1, output0, h1_mem, h1_spike)
                if self.rnn_shape[0] > 0:
                    h1_y             = self.rnn_update(self.rnn_fc1, self.rnn_fv1, output0, h1_y)
                output1          = torch.cat((h1_spike, h1_y), dim=1)
                if self.snn_shape[1] > 0:
                    h2_mem, h2_spike = self.snn_update(self.snn_fc2, output1, h2_mem, h2_spike)
                if self.rnn_shape[1] > 0:
                    h2_y             = self.rnn_update(self.rnn_fc2, self.rnn_fv2, output1, h2_y)
                h2_sum_spike     = h2_sum_spike + h2_spike
            output2 = torch.cat((h2_sum_spike / time_window, h2_y), dim=1)
        else:
            for step in range(time_window):
                output0 = buf_input[:,  step,:]
                output0 = output0.view(batch_size, -1)
                if self.snn_shape[0] > 0:
                    h1_mem, h1_spike = self.snn_update(self.snn1, output0, h1_mem, h1_spike)
                if self.rnn_shape[0] > 0:
                    h1_y             = self.rnn_union_update(self.rnn1, output0, h1_y)
                output1          = torch.cat((h1_spike, h1_y), dim=1)

                if self.snn_shape[1] > 0:
                    h2_mem, h2_spike = self.snn_update(self.snn2, output1, h2_mem, h2_spike)
                if self.rnn_shape[1] > 0:
                    h2_y             = self.rnn_union_update(self.rnn2, output1, h2_y)
                h2_sum_spike     = h2_sum_spike + h2_spike
            output2 = torch.cat((h2_sum_spike / time_window, h2_y), dim=1)                 
        
        dout = output2

        return self.all_fc(dout), (h1_y, h2_y)
    


###############################################################################
# Parse arguments
###############################################################################

parser = argparse.ArgumentParser(description='HNN Model on MNIST Dataset')

parser.add_argument('--data', type=str,
                    default='../../data/MNIST/',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='hybrid', help='type of network (rnn, snn, ffs[directly-hybrid], hybrid)')

parser.add_argument('--mode', type=str, default='train1', help='type of operation (train1, test1, train2, test2)')
# Activate parameters
parser.add_argument('--batch_size', type=int, default=100, metavar='N', help='batch size')
parser.add_argument('--stage1_epochs', type=int, default=150, help='upper epoch limit')
parser.add_argument('--stage2_epochs', type=int, default=150, help='upper epoch limit')
parser.add_argument('--stage1_lr', type=float, default=3e-4, help='upper epoch limit')
parser.add_argument('--stage2_lr', type=float, default=3e-4, help='upper epoch limit')

parser.add_argument('--ratio', type=float, default=0.5, help='snn ratio of hybird network')

# Default parameters
parser.add_argument('--nhid', type=int, default=100, help='number of hidden units per layer')
parser.add_argument('--clip', type=float, default=0.25, help='gradient clipping')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout applied to layers (0 = no dropout)')
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

traindataset = torchvision.datasets.MNIST(root=args.data, train=True, download=True, transform=transforms.ToTensor())
trainloader  = torch.utils.data.DataLoader(traindataset, batch_size=args.batch_size, shuffle=True, num_workers=0)

testset      = torchvision.datasets.MNIST(root=args.data, train=False, download=True,  transform=transforms.ToTensor())
testloader   = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False, num_workers=0)

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

#criterion = nn.MSELoss()
criterion = nn.CrossEntropyLoss()

def train(arg_model, optimizers, epoch, logger):
    arg_model.train()
    train_loss = 0
    correct    = 0
    total      = 0
    start_time = time.time()
    hidden = arg_model.init_hidden(args.batch_size)

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs  = inputs.to(device)
        targets = targets.to(device)

        hidden = repackage_hidden(hidden)
        arg_model.zero_grad()
        outputs, hidden = arg_model(inputs, hidden)
        
        # MSE Loss (archived)
        #labels_     = torch.zeros(args.batch_size, args.nclass, device = device).scatter_(1, targets.view(-1, 1), 1)
        #target_loss = criterion(outputs, labels_)

        target_loss = criterion(outputs, targets)

        both_loss   = target_loss.to(device)
        both_loss.backward()
        torch.nn.utils.clip_grad_norm_(arg_model.parameters(), args.clip)

        optimizers.step()
        optimizers.zero_grad()
        
        train_loss  += target_loss.item()
        _, predicted = outputs.max(1)
        total       += targets.size(0)
        correct     += predicted.eq(targets).sum().item()

        if batch_idx % args.log_interval == 0 and batch_idx > 0:
            cur_loss = train_loss / args.log_interval
            elapsed  = time.time() - start_time
            logger.info('| epoch {:3d} | {:5d}/{:5d} batches | {:5.2f} ms/batch '
                        '| loss {:.5f} | acc {:.3f}%'.format(
                            epoch, batch_idx, 600, elapsed * 1000 / args.log_interval, cur_loss, 100.0 * correct / total)
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

    hidden = arg_model.init_hidden(args.batch_size)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs = inputs.to(device)
            targets    = targets.to(device)            
            outputs, hidden = arg_model(inputs, hidden)
            loss        = criterion(outputs, targets)
            test_loss   += loss.item()
            _, predicted = outputs.max(1)
            total       += targets.size(0)
            correct     += predicted.eq(targets).sum().item()
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
                'rnn_shape' : arg_model.rnn_shape, 
                'snn_shape' : arg_model.snn_shape,
                'snn_func'  : arg_model.func,
                'test_acc_record' : test_acc_record
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, './checkpoint/' + arg_model_name + ".t7")
    test_trained_model(best_model, logger)
    return best_model

def test_trained_model(arg_model, logger):
    test_acc, test_loss = evaluate(arg_model)
    logger.info('=' * 83)
    logger.info('| Performance on Test Set | test loss {:.5f} | test acc {:.5f}'.format(
        test_loss, test_acc))
    logger.info('=' * 83)
      
def train_origin_model(model_name, logger):
    if args.model == "rnn":
        rnn_shape = [args.nhid, args.nhid]
        snn_shape = [0, 0]
    elif args.model == "snn":
        rnn_shape = [0, 0]
        snn_shape = [args.nhid, args.nhid]
    elif args.model == "hybrid":
        rnn_shape = [args.nhid, args.nhid]
        snn_shape = [args.nhid, args.nhid]
    else:   # learning from scratch
        rnn_shape = [int((1 - args.ratio) * args.nhid), int((1 - args.ratio) * args.nhid)]
        snn_shape = [int(args.ratio * args.nhid),       int(args.ratio * args.nhid)]

    logger.info("model with rnn_shape: [{:3d}, {:3d}]".format(rnn_shape[0], rnn_shape[1]))
    logger.info("model with snn_shape: [{:3d}, {:3d}]".format(snn_shape[0], snn_shape[1]))

    origin_model = HNNModel(args.nclass, 28, rnn_shape=rnn_shape, snn_shape=snn_shape, dropout=args.dropout, device="cuda").to(device)
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


def compute_effective_indices(arg_model, logger, hessian_mode="trace"):
    assert args.model == "hybrid"
    from hessian_pruner import HessianPruner
    if not os.path.isdir('traces'):
                os.mkdir('traces')
    # todo: func!!!
    trace_file_name = "./traces/" + "trace" + "_" + "hybrid" + "_"  + str(args.seed) + ".npy"
    pruner = HessianPruner(arg_model, trace_file_name, hessian_mode=hessian_mode)
    eff_dict = pruner.make_pruned_model(trainloader, criterion, device, args.ratio, args.seed)

    print_eff_index("rnn1", sorted(eff_dict["rnn1"]), logger)
    print_eff_index("rnn2", sorted(eff_dict["rnn2"]), logger)
    print_eff_index("snn1", sorted(eff_dict["snn1"]), logger)
    print_eff_index("snn2", sorted(eff_dict["snn2"]), logger)

    logger.info("rnn_layer[0]: " + str(len(eff_dict["rnn1"])))
    logger.info("rnn_layer[1]: " + str(len(eff_dict["rnn2"])))
    logger.info("snn_layer[0]: " + str(len(eff_dict["snn1"])))
    logger.info("snn_layer[1]: " + str(len(eff_dict["snn2"])))
    
    return eff_dict



def logger_generation(file_name):
    if not os.path.isdir('log'):
        os.mkdir('log')
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
    tmp_rnn_shape = ckpts["rnn_shape"]
    tmp_snn_shape = ckpts["snn_shape"]
    tmp_nclass    = 10
    tmp_ninp      = 28
    # todo: func!!!
    tmp_func      = "window"
    tmp_model     = HNNModel(nclass=tmp_nclass, ninp=tmp_ninp, rnn_shape=tmp_rnn_shape, snn_shape=tmp_snn_shape, dropout=args.dropout, device=device, func=tmp_func, union=make_union).to(device)
    if make_union == False:
        tmp_model.load_state_dict(weights_dict)
    else:
        tmp_model.load_union_state(weights_dict)
    return tmp_model 

def weight_shrink(arg_weight, eff_row = None, row_size = None, eff_col = None, col_size = None):
    assert arg_weight is not None
    tmp_weight = copy.deepcopy(arg_weight).to("cpu")
    if row_size is not None:
        assert tmp_weight.shape[0] == row_size
        left_matrix = torch.zeros(len(eff_row), row_size)
        for per_row in range(len(eff_row)):
            left_matrix[per_row][eff_row[per_row]] = 1
        cc = torch.mm(left_matrix, tmp_weight)
        tmp_weight = copy.deepcopy(torch.mm(left_matrix, tmp_weight))
    if col_size is not None:
        assert tmp_weight.shape[1] == col_size
        right_matrix = torch.zeros(col_size, len(eff_col))
        for per_col in range(len(eff_col)):
            right_matrix[eff_col[per_col]][per_col] = 1
        tmp_weight = copy.deepcopy(torch.mm(tmp_weight, right_matrix))
    return tmp_weight




def shrink(arg_model, index_dict):
    pruned_model = copy.deepcopy(arg_model).to("cpu")
    old_dict = arg_model.state_dict()
    #en_w     = old_dict["encoder.weight"] # (raw_input_length, embedded_size)
    de_w     = old_dict["all_fc.weight"] # (raw_output_length, rnn_layer[1] + snn_layer[1])
    de_b     = old_dict["all_fc.bias"]   # (raw_output_length)
    
    snn_fc1     = old_dict["snn_fc1"]     # (embbeded_size, snn_layer[0])
    rnn_fc1     = old_dict["rnn_fc1"]     # (embbeded_size, rnn_layer[0])
    rnn_fv1     = old_dict["rnn_fv1"]     # (rnn_layer[0], rnn_layer[0])

    snn_fc2     = old_dict["snn_fc2"]     # (snn_layer[0] + rnn_layer[0], snn_layer[1]) 
    rnn_fc2     = old_dict["rnn_fc2"]     # (snn_layer[0] + rnn_layer[0], rnn_layer[1])
    rnn_fv2     = old_dict["rnn_fv2"]     # (rnn_layer[1], rnn_layer[1])

    snn_fc1_eff_col = sorted(index_dict["snn1"])
    rnn_fc1_eff_col = sorted(index_dict["rnn1"])
    rnn_fv1_eff_col = sorted(index_dict["rnn1"])
    rnn_fv1_eff_row = sorted(index_dict["rnn1"])

    snn_fc2_eff_col = sorted(index_dict["snn2"])
    rnn_fc2_eff_col = sorted(index_dict["rnn2"])
    rnn_fv2_eff_col = sorted(index_dict["rnn2"])
    rnn_fv2_eff_row = sorted(index_dict["rnn2"])

    # mention!
    incre1 = arg_model.snn_shape[0]
    tmp1 = [index + incre1 for index in index_dict["rnn1"]]
    snn_fc2_eff_row = sorted(index_dict["snn1"] + tmp1)
    rnn_fc2_eff_row = snn_fc2_eff_row

    incre2 = arg_model.snn_shape[1]
    tmp2 = [index + incre2 for index in index_dict["rnn2"]]
    de_w_eff_col = sorted(index_dict["snn2"] + tmp2)

    snn_fc1_new = weight_shrink(snn_fc1, eff_row=None, row_size=None,
                                         eff_col=snn_fc1_eff_col, col_size=arg_model.snn_shape[0])
    rnn_fc1_new = weight_shrink(rnn_fc1, eff_row=None, row_size=None,
                                         eff_col=rnn_fc1_eff_col, col_size=arg_model.rnn_shape[0])
    rnn_fv1_new = weight_shrink(rnn_fv1, eff_row=rnn_fv1_eff_row, row_size=arg_model.rnn_shape[0],
                                         eff_col=rnn_fv1_eff_col, col_size=arg_model.rnn_shape[0])
    
    snn_fc2_new = weight_shrink(snn_fc2, eff_row=snn_fc2_eff_row, row_size=arg_model.rnn_shape[0] + arg_model.snn_shape[0],
                                         eff_col=snn_fc2_eff_col, col_size=arg_model.snn_shape[1])
    rnn_fc2_new = weight_shrink(rnn_fc2, eff_row=rnn_fc2_eff_row, row_size=arg_model.rnn_shape[0] + arg_model.snn_shape[0],
                                         eff_col=rnn_fc2_eff_col, col_size=arg_model.rnn_shape[1])
    rnn_fv2_new = weight_shrink(rnn_fv2, eff_row=rnn_fv2_eff_row, row_size=arg_model.rnn_shape[1],
                                         eff_col=rnn_fv2_eff_col, col_size=arg_model.rnn_shape[1])    

    de_w_new    = weight_shrink(de_w, eff_row=None, row_size=None, eff_col=de_w_eff_col, 
                                         col_size=arg_model.rnn_shape[1] + arg_model.snn_shape[1])


    pruned_model.rnn_shape = [len(index_dict["rnn1"]), len(index_dict["rnn2"])]
    pruned_model.snn_shape = [len(index_dict["snn1"]), len(index_dict["snn2"])]
    pruned_model.val_loss_record = []

    pruned_model.snn_fc1.data = copy.deepcopy(snn_fc1_new).to("cuda")
    pruned_model.snn_fc2.data = copy.deepcopy(snn_fc2_new).to("cuda")
    pruned_model.rnn_fc1.data = copy.deepcopy(rnn_fc1_new).to("cuda")
    pruned_model.rnn_fc2.data = copy.deepcopy(rnn_fc2_new).to("cuda")
    pruned_model.rnn_fv1.data = copy.deepcopy(rnn_fv1_new).to("cuda")
    pruned_model.rnn_fv2.data = copy.deepcopy(rnn_fv2_new).to("cuda")
    pruned_model.all_fc.weight.data = copy.deepcopy(de_w_new).to("cuda")
    pruned_model.all_fc.bias.data = copy.deepcopy(de_b).to("cuda")

    del snn_fc1_new, snn_fc2_new
    del rnn_fc1_new, rnn_fc2_new
    del rnn_fv1_new, rnn_fv2_new
    del de_w_new, de_b
    pruned_model.to("cuda")
    return pruned_model


def get_pruned_model(model_name, logger, union=False, hessian_mode="trace"):
    union_model = load_model(model_name, make_union=union)
    test_trained_model(union_model, logger)
    eff_dict    = compute_effective_indices(union_model, logger, hessian_mode=hessian_mode)
    union_model = union_model.cpu()
    del union_model
    trained_model = load_model(model_name, make_union=False)
    pruned_model = shrink(trained_model, eff_dict)
    test_trained_model(pruned_model, logger)
    trained_model = trained_model.cpu()
    del trained_model
    return pruned_model

assert args.model in ["rnn", "snn", "ffs", "hybrid"]

dataset_name = "mnist"


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
    print(trained_model.rnn_shape, trained_model.snn_shape)
    test_trained_model(trained_model, test1_logger)

elif args.mode == "train2":
    assert args.model == "hybrid"
    if args.model == "hybrid":
        input_model_name  = dataset_name + "_" + args.model + "_" + str(args.seed)
        pruned_model_name = dataset_name + "_" + args.model + "_" + str(args.seed) + "_" + str(args.ratio)
        logfile_name = "train" + "_" + pruned_model_name
        train2_logger = logger_generation(logfile_name)
        pruned_model = get_pruned_model(input_model_name, train2_logger, union=True) # caution!
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
