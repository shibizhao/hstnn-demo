import os
import torch

base_config = [
    {"input_channels": 2,   "cnn_channels": 128, "conv_kernel_size": 3, "pool_kernel_size": 2, "conv_stride": 1, "pool_stride": 2, "fc_padding": 1, "fv_padding": 1, "avr_pool": True},
    {"input_channels": 128, "cnn_channels": 256, "conv_kernel_size": 3, "pool_kernel_size": 2, "conv_stride": 1, "pool_stride": 2, "fc_padding": 1, "fv_padding": 1, "avr_pool": True},
    {"input_channels": 256, "cnn_channels": 384, "conv_kernel_size": 3, "pool_kernel_size": 2, "conv_stride": 1, "pool_stride": 2, "fc_padding": 1, "fv_padding": 1, "avr_pool": True}
]

buffer_config = [
    {"output_size": 32, "pooled_size": 16},
    {"output_size": 16, "pooled_size": 8},
    {"output_size": 8, "pooled_size": 4}
]

linear_config = [
    {"input_channels": 384 * 4 * 4, "output_channels": 256},
    {"input_channels": 256, "output_channels": 11},
]