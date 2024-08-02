import torch
import torch.nn as nn
from collections import OrderedDict
from utils.kfac_utils import fetch_mat_weights
# from utils.common_utils import (tensor_to_list, PresetLRScheduler)
from utils.prune_utils import (filter_indices,
                               filter_indices_ni,
                               get_threshold,
                               update_indices,
                               normalize_factors,
                               prune_model_ni)
# from utils.network_utils import stablize_bn
from tqdm import tqdm

from hessian_fact import get_trace_hut
from pyhessian.hessian import hessian
from pyhessian.utils import group_product, group_add, normalization, get_params_grad, hessian_vector_product, orthnormal

import numpy as np
import time
import scipy.linalg
import os.path
from os import path


class HessianPruner:
        def __init__(self,
                     model,
                     trace_file_name,
                     fix_layers=0,
                     hessian_mode='trace'):
            self.iter = 0
            #self.known_modules = {'Linear', 'Conv2d'}
            self.modules = []
            self.model = model
            self.fix_layers = fix_layers
            
            self.W_pruned = {}
            self.S_l = None

            self.hessian_mode = hessian_mode
            self.trace_file_name = trace_file_name

            self.importances = {}
            self._inversed = False
            self._cfgs = {}
            self._indices = {}


        # top function
        def make_pruned_model(self, dataloader, criterion, device, snn_ratio, seed, is_loader=False, normalize=True, re_init=False, n_v=300):
            self.snn_ratio = snn_ratio # use for some special case, particularly slq_full, slq_layer
            self.seed = seed
            self._prepare_model()
            self.mask_list = self._compute_hessian_importance(dataloader, criterion, device, is_loader, n_v=n_v)
            print("Finished Hessian Importance Computation!")
            return self.mask_list

        def _prepare_model(self):
            count = 0
            for it in self.model.named_parameters():
                if it[0].find("all_fc") >= 0:
                    continue
                if it[0].find("wh") >= 0:
                    continue
                if it[0].find("decoder") >= 0:
                    continue
                if it[0].find("fv") >= 0:
                    continue
                if it[0].find("encoder") >= 0:
                    continue
                self.modules.append(it)


        def _compute_hessian_importance(self, dataloader, criterion, device, is_loader, n_v=300):
            print("is_loader", is_loader)
            ###############
            # Here, we use the fact that Conv does not have bias term
            ###############
            if self.hessian_mode == 'trace':
                for k, v in self.model.named_parameters():
                    if k.find("all_fc") >= 0:
                        v.requires_grad = False
                    elif k.find("wh") >= 0:
                        v.requires_grad = False
                    elif k.find("decoder") >= 0:
                        v.requires_grad = False
                    elif k.find("fv") >= 0:
                        v.requires_grad = False
                    elif k.find("encoder") >= 0:
                        v.requires_grad = False
                    else:
                        print(k, v.requires_grad)

                trace_dir = self.trace_file_name
                print(trace_dir)
                if  os.path.exists(trace_dir):
                    import numpy as np
                    print(f"Loading trace from {trace_dir}")
                    results = np.load(trace_dir, allow_pickle=True)
                else:
                    import numpy as np
                    results = get_trace_hut(self.model, dataloader, criterion, n_v=n_v, loader=is_loader, channelwise=True, layerwise=False)
                    np.save(self.trace_file_name, results)

                for m in self.model.parameters():
                    m.requires_grad = True

                channel_trace, weighted_trace = [], []

                for k, layer in enumerate(results):
                    # print(k, layer)
                    channel_trace.append(torch.zeros(len(layer)))
                    weighted_trace.append(torch.zeros(len(layer)))
                    for cnt, channel in enumerate(layer):
                        #print(cnt, channel.shape, len(layer))
                        channel_trace[k][cnt] = sum(channel) / len(channel)
                #for i in channel_trace:
                    # print(len(i))

                # print(len(results), self.model.parameters())

                for k, mod in enumerate(self.modules):
                    tmp = []
                    m = mod[0]
                    import copy 
                    cur_weight = copy.deepcopy(mod[1].data)
                    dims = len(list(cur_weight.size()))
                    if dims == 2:
                        cur_weight = cur_weight.permute(1, 0)
                    elif dims == 3:
                        cur_weight = cur_weight.permute(2, 0, 1)
                    for cnt, channel in enumerate(cur_weight):
                        tmp.append( (channel_trace[k][cnt] * channel.detach().norm()**2 / channel.numel()).cpu().item())
                    print(m, len(tmp))
                    self.importances[str(m)] = (tmp, len(tmp))
                    #self.W_pruned[m] = fetch_mat_weights(m, False)

            else:
                print("Unknown Mode")
                assert False

            tmp_imp_list = list(self.importances.items())   

            rnn_list = [None, None]
            snn_list = [None, None]

            for unit in tmp_imp_list:
                if unit[0].find("rnn") >= 0 or unit[0].find("lstm") >= 0:
                    if unit[0].find("1") >= 0:
                        rnn_list[0] = unit[1][0]
                    else:
                        assert unit[0].find("2") >= 0
                        rnn_list[1] = unit[1][0]
                elif unit[0].find("snn") >= 0:
                    if unit[0].find("1") >= 0:
                        snn_list[0] = unit[1][0]
                    else:
                        assert unit[0].find("2") >= 0
                        snn_list[1] = unit[1][0]
                else:
                    continue

            rnn_shape = [len(rnn_list[0]), len(rnn_list[1])]
            snn_shape = [len(snn_list[0]), len(snn_list[1])]

            rnn_tuple_list = []
            snn_tuple_list = []

            for no in range(len(rnn_list[0])):
                rnn_tuple_list.append((no, rnn_list[0][no]))
            for no in range(len(rnn_list[1])):
                rnn_tuple_list.append((no + rnn_shape[0], rnn_list[1][no]))
            
            for no in range(len(snn_list[0])):
                snn_tuple_list.append((no, snn_list[0][no]))
            for no in range(len(snn_list[1])):
                snn_tuple_list.append((no + snn_shape[0], snn_list[1][no]))

            sorted_rnn_list = sorted(rnn_tuple_list, key=lambda x:x[1])#, reverse=True)
            sorted_snn_list = sorted(snn_tuple_list, key=lambda x:x[1])#, reverse=True)

            sorted_rnn_list.reverse()
            sorted_snn_list.reverse()
            
            del rnn_list, snn_list, rnn_tuple_list, snn_tuple_list

            eff_rnns_number = int((rnn_shape[0] + rnn_shape[1]) * (1.0 - self.snn_ratio))
            eff_snns_number = int((snn_shape[0] + snn_shape[1]) * (self.snn_ratio))

            rnn_layer_util = [False, False]
            snn_layer_util = [False, False]

            # check whether at least one neuron(rnn or snn) exists in every layer
            for idx in range(0, eff_rnns_number):
                if sorted_rnn_list[idx][0] >= rnn_shape[0]:
                    rnn_layer_util[1] = True
                else:
                    rnn_layer_util[0] = True
            
            for idx in range(0, eff_snns_number):
                if sorted_snn_list[idx][0] >= snn_shape[0]:
                    snn_layer_util[1] = True
                else:
                    snn_layer_util[0] = True
            
            # fix the structure
            def not_in_one_layer(idx1, idx2, thres):
                return (idx1 < thres and idx2 >= thres) or (idx2 < thres and idx1 >= thres)
            
            eff_rnns_list = []
            for idx in range(0, eff_rnns_number):
                eff_rnns_list.append(sorted_rnn_list[idx][0])
            
            if rnn_layer_util[0] is False or rnn_layer_util[1] is False:
                last_one = eff_rnns_list[-1]
                for idx in range(eff_rnns_number, rnn_shape[0] + rnn_shape[1]):
                    curr_one = sorted_rnn_list[idx][0]
                    if not_in_one_layer(last_one, curr_one, rnn_shape[0]) is True:
                        eff_rnns_list[-1] = curr_one
                        break

            eff_snns_list = []
            for idx in range(0, eff_snns_number):
                eff_snns_list.append(sorted_snn_list[idx][0])              

            if snn_layer_util[0] is False or snn_layer_util[1] is False:
                last_one = eff_snns_list[-1]
                for idx in range(eff_snns_number, snn_shape[0] + snn_shape[1]):
                    curr_one = sorted_snn_list[idx][0]
                    if not_in_one_layer(last_one, curr_one, snn_shape[0]) is True:
                        eff_snns_list[-1] = curr_one
                        break

            del rnn_layer_util, snn_layer_util


            # output
            eff_dict = {}
            eff_dict["rnn1"] = []
            eff_dict["rnn2"] = []
            eff_dict["snn1"] = []
            eff_dict["snn2"] = []

            for item in eff_rnns_list:
                if item < rnn_shape[0]:
                    eff_dict["rnn1"].append(item)
                else:
                    eff_dict["rnn2"].append(item - rnn_shape[0])
            
            for item in eff_snns_list:
                if item < snn_shape[0]:
                    eff_dict["snn1"].append(item)
                else:
                    eff_dict["snn2"].append(item - snn_shape[0])
            # print(sorted_rnn_list)
            # print(eff_dict)
            # print(sorted_snn_list)
            return eff_dict
