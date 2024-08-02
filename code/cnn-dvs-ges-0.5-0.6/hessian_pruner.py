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
                     hessian_mode='trace',
                     neuron_nums_dict = {"rcnn": -1, "scnn": -1}):
            self.iter = 0
            #self.known_modules = {'Linear', 'Conv2d'}
            self.modules = {}
            self.model = model
            self.fix_layers = fix_layers
            
            self.known_modules = {'Conv2d'}

            self.W_pruned = {}
            self.S_l = None

            self.hessian_mode = hessian_mode
            self.trace_file_name = trace_file_name

            self.importances = {}
            self._inversed = False
            self._cfgs = {}
            self._indices = {}

            self.neuron_nums_dict = neuron_nums_dict

            assert neuron_nums_dict["rcnn"] >= 0 and neuron_nums_dict["scnn"] >= 0


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
            for name, module in self.model.named_modules():
                classname = module.__class__.__name__
                if classname in self.known_modules and name.find("fv") < 0 and name.find("linear_layers") < 0:
                    self.modules[name] = module
                    count += 1                


        def _compute_hessian_importance(self, dataloader, criterion, device, is_loader, n_v=300):
            print("is_loader", is_loader)
            ###############
            # Here, we use the fact that Conv does not have bias term
            ###############
            if self.hessian_mode == 'trace':
                for k, v in self.model.named_parameters():
                    if k.find("bias") >= 0:
                        v.requires_grad = False
                    elif k.find("wh") >= 0:
                        v.requires_grad = False
                    elif k.find("decoder") >= 0:
                        v.requires_grad = False
                    elif k.find("fv") >= 0:
                        v.requires_grad = False
                    elif k.find("linear_layers") >= 0:
                        v.requires_grad = False
                    else:
                        print(k, v.requires_grad)

                trace_dir = self.trace_file_name
                print(trace_dir)
                if os.path.exists(trace_dir):
                    import numpy as np
                    print(f"Loading trace from {trace_dir}")
                    results = np.load(trace_dir, allow_pickle=True).item()
                else:
                    import numpy as np
                    pname, results = get_trace_hut(self.model, dataloader, criterion, n_v=n_v, loader=is_loader, channelwise=True, layerwise=False)

                    my_dict = {}
                    for i in range(len(results)):
                        my_dict[pname[i]] = results[i]

                    results = my_dict

                    np.save(self.trace_file_name, my_dict)


                for m in self.model.parameters():
                    m.requires_grad = True

                channel_trace, weighted_trace = [], []

                idx = 0
                for k, layer in results.items():
                    # print(k, layer)
                    channel_trace.append(torch.zeros(len(layer)))
                    weighted_trace.append(torch.zeros(len(layer)))
                    for cnt, channel in enumerate(layer):
                        channel_trace[idx][cnt] = sum(channel) / len(channel)
                    idx += 1

                idx = 0
                for k, mod in self.modules.items():
                    tmp = []
                    import copy 
                    cur_weight = copy.deepcopy(mod.weight.data)
                    for cnt, channel in enumerate(cur_weight):
                        tmp.append( (channel_trace[idx][cnt] * channel.detach().norm()**2 / channel.numel()).cpu().item())
                    print(k, len(tmp))
                    self.importances[str(k)] = (tmp, len(tmp))
                    idx += 1
                    #self.W_pruned[m] = fetch_mat_weights(m, False)

            else:
                print("Unknown Mode")
                assert False

            importance_dict = {}

            selected_neurons = {"rcnn": [], "scnn": []}

            overall_pools = {"rcnn": [], "scnn": []}

            grouped_neurons = {}
            
            for k, v in self.importances.items():
                importance_dict[k] = [(v[0][i], i, k) for i in range(len(v[0]))]

            
            for k in importance_dict.keys():
                vec = sorted(importance_dict[k], key=lambda x:x[0], reverse=True)
                if k.find("rcnn") >= 0:
                    selected_neurons["rcnn"].append(vec[0])
                    overall_pools["rcnn"] += vec[1:]
                elif k.find("scnn") >= 0:
                    selected_neurons["scnn"].append(vec[0])
                    overall_pools["scnn"] += vec[1:]


            for key in overall_pools.keys():
                vec = sorted(overall_pools[key], key=lambda x:x[0], reverse=True)
                needed_neurons = self.neuron_nums_dict[key] - len(selected_neurons[key])
                selected_neurons[key] += vec[0 : needed_neurons]
            
            for key in self.importances.keys():
                grouped_neurons[key] = []

            for key, vlist in selected_neurons.items():
                for neuron in vlist:
                    # neuron[0]: importance, neuron[1]: index in its layer, neuron[2]: layer name
                    grouped_neurons[neuron[2]].append(neuron[1])




            for key, value in grouped_neurons.items():
                value.sort()
                print(key, len(value), max(value))
                    
            
            




            return grouped_neurons
