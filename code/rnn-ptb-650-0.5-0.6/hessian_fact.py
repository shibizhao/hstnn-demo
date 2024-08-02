import torch
import math
import copy
from torch.autograd import Variable
import numpy as np
from progress.bar import Bar

import copy
import time


def group_product(xs, ys):
    """
    the inner product of two lists of variables xs,ys
    :param xs:
    :param ys:
    :return:
    """
    return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])


def group_add(params, update, alpha=1):
    """
    params = params + update*alpha
    :param params: list of variable
    :param update: list of data
    :return:
    """
    for i,p in enumerate(params):
        params[i].data.add_(update[i] * alpha)
    return params


def normalization(v):
    """
    normalization of a list of vectors
    return: normalized vectors v
    """
    s = group_product(v,v)
    s = s ** 0.5
    s = s.cpu().item()
    v = [vi / (s + 1e-6) for vi in v]
    return v


def get_params_grad(model):
    """
    get model parameters and corresponding gradients
    """
    params = []
    grads = []
    for param in model.parameters():
        if not param.requires_grad:
            continue
        params.append(param)
        grads.append(0. if param.grad is None else param.grad + 0.)
    return params, grads


def hessian_vector_product(gradsH, params, v, stop_criterion=False):
    """
    compute the hessian vector product of Hv, where
    gradsH is the gradient at the current point,
    params is the corresponding variables,
    v is the vector.
    """
    hv = torch.autograd.grad(gradsH, params, grad_outputs=v, only_inputs=True, retain_graph = not stop_criterion)
    return hv


def orthnormal(w, v_list):
    for v in v_list:
        w = group_add(w, v, alpha=-group_product(w, v))
    return normalization(w)

def hessian_get_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].view(-1)
    return data, target


def get_trace_hut(model, data, criterion, n_v, batch_size, bptt, ntokens, loader, cuda = True, channelwise = False, layerwise = False):
    """
    compute the trace of hessian using Hutchinson's method
    """
    
    print("batch_size: {:5d} | bptt: {:5d} | ntokens: {:5d}".format(batch_size, bptt, ntokens))
    
    assert not (channelwise and layerwise)
    # if loader:
    inputs, targets = hessian_get_batch(data, 0, bptt)
    # else:
    #     inputs, targets = data

    if cuda:
        inputs, targets = inputs.cuda(), targets.cuda()
        device = 'cuda'
    else:
        device = 'cpu'
    model.eval()
    
    hidden = model.init_hidden(batch_size)
    outputs, hidden = model(inputs, hidden)

    loss = criterion(outputs.view(-1, ntokens), targets)
    loss.backward(create_graph = True)

    params, gradsH = get_params_grad(model)

    for p in params:
        print(p.shape, p.size(-1))
    if channelwise:
        trace_vhv = [[[] for c in range(p.size(-1))] for p in params]
    elif layerwise:
        trace_vhv = [[] for p in params]
    else:
        trace_vhv = []

    # print(np.array(trace_vhv).shape)


    bar = Bar('Computing trace', max=n_v)

    # total_grad_out = []
    # total_grad_in = []

    # def hook_fn_backward(module, grad_input, grad_output):
    #     total_grad_in.append(grad_input)
    #     total_grad_out.append(grad_output)

    # for name, module in model.named_children():
    #     module.register_backward_hook(hook_fn_backward)

    for i in range(n_v):
        start_time = time.time()
        bar.suffix = f'({i + 1}/{n_v}) |ETA: {bar.elapsed_td}<{bar.eta_td}'
        bar.next()
        v = [torch.randint_like(p, high = 2, device = device).float() * 2 - 1 for p in params]
        # if loader:
        #     print("loader: ", loader)
        #     THv = [torch.zeros(p.size()).to(device) for p in params]
        #     hidden = model.init_hidden(100)
        #     for inputs, targets in data:
        #         inputs, targets = inputs.to(device), targets.to(device)
                
        #         outputs, hidden = model(inputs, hidden)
        #         labels_ =  torch.zeros(100, 10, device = device).scatter_(1, targets.view(-1, 1), 1).to(device)
        #         model.zero_grad()
        #         outputs = model(inputs)
        #         loss = criterion(outputs, targets)
        #         loss.backward(create_graph = True)

        #         params, gradsH = get_params_grad(model)
        #         Hv = torch.autograd.grad(gradsH, params, grad_outputs = v, only_inputs = True, retain_graph = False)
        #         # remenber to normalize over dummy-batch
        #         THv = [THv1 + Hv1/float(len(data)) + 0. for THv1, Hv1 in zip(THv, Hv)]
        #     Hv = THv
        # else:

        Hv = hessian_vector_product(gradsH, params, v, stop_criterion= (i==(n_v-1)))

        Hv = [Hvi.detach().cpu() for Hvi in Hv]
        v = [vi.detach().cpu() for vi in v]

        with torch.no_grad():
            import copy
            if channelwise:
                for Hv_i in range(len(Hv)):
                    for channel_i in range(Hv[Hv_i].size(-1)):
                        dims = len(list(Hv[Hv_i].size()))
                        tmp_hv = copy.deepcopy(Hv[Hv_i])
                        tmp_v  = copy.deepcopy(v[Hv_i])
                        if dims == 2:   #SNN/RNN
                            tmp_hv = tmp_hv.permute(1, 0)
                            tmp_v  = tmp_v.permute(1, 0)
                        elif dims == 3: #LSTM
                            tmp_hv = tmp_hv.permute(2, 0, 1)
                            tmp_v  = tmp_v.permute(2, 0, 1)                            
                        trace_vhv[Hv_i][channel_i].append(tmp_hv[channel_i].flatten().dot(tmp_v[channel_i].flatten()).item())
                        del tmp_hv
                        del tmp_v   
            elif layerwise:
                for Hv_i in range(len(Hv)):
                    trace_vhv[Hv_i].append(Hv[Hv_i].flatten().dot(v[Hv_i].flatten()).item())
            else:
                trace_vhv.append(group_product(Hv, v).item())
    bar.finish()
    return trace_vhv
