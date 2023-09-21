#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import random
import numpy as np
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from math import floor, ceil
from datetime import datetime

from contextlib import contextmanager

logger = logging.getLogger(__name__)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    return None



def init_logger(fl_prefix = None, drc_log = "logs", returns_timestamp = False):
    # Create a timestamp to name the log file...
    now = datetime.now()
    timestamp = now.strftime("%Y_%m%d_%H%M_%S")

    # Set up the log file...
    # ...filename
    fl_log = f"{timestamp}.log"
    if fl_prefix is not None: fl_log = f"{fl_prefix}.{fl_log}"

    # ...path
    os.makedirs(drc_log, exist_ok = True)
    path_log = os.path.join(drc_log, fl_log)

    # Config logging behaviors
    logging.basicConfig( filename = path_log,
                         filemode = 'w',
                         format="%(asctime)s %(levelname)s %(name)s\n%(message)s",
                         datefmt="%m/%d/%Y %H:%M:%S",
                         level=logging.INFO, )
    logger = logging.getLogger(__name__)

    return timestamp if returns_timestamp else None




class MetaLog:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        for k, v in kwargs.items(): setattr(self, k, v)


    def report(self):
        logger.info(f"___/ MetaLog \___")
        for k, v in self.__dict__.items():
            if k == 'kwargs': continue
            logger.info(f"KV - {k:16s} : {v}")




def split_dataset(dataset_list, fracA, seed = None):
    ''' Split a dataset into two subsets A and B by user-specified fraction.
    '''
    # Set seed for data spliting...
    if seed is not None:
        random.seed(seed)

    # Indexing elements in the dataset...
    size_dataset = len(dataset_list)
    idx_dataset = range(size_dataset)

    # Get the size of the dataset and the subset A...
    size_fracA   = int(fracA * size_dataset)

    # Randomly choosing examples for constructing subset A...
    idx_fracA_list = random.sample(idx_dataset, size_fracA)

    # Obtain the subset B...
    idx_fracB_list = set(idx_dataset) - set(idx_fracA_list)
    idx_fracB_list = sorted(list(idx_fracB_list))

    fracA_list = [ dataset_list[idx] for idx in idx_fracA_list ]
    fracB_list = [ dataset_list[idx] for idx in idx_fracB_list ]

    return fracA_list, fracB_list




def split_list_into_chunk(input_list, max_num_chunk = 2):

    chunk_size = len(input_list) // max_num_chunk + 1

    size_list = len(input_list)

    chunked_list = []
    for idx_chunk in range(max_num_chunk):
        idx_b = idx_chunk * chunk_size
        idx_e = idx_b + chunk_size
        ## if idx_chunk == max_num_chunk - 1: idx_e = len(input_list)
        if idx_e >= size_list: idx_e = size_list

        seg = input_list[idx_b : idx_e]
        chunked_list.append(seg)

        if idx_e == size_list: break

    return chunked_list




def split_dict_into_chunk(input_dict, max_num_chunk = 2):

    chunk_size = len(input_dict) // max_num_chunk + 1

    size_dict = len(input_dict)
    kv_iter   = iter(input_dict.items())

    chunked_dict_in_list = []
    for idx_chunk in range(max_num_chunk):
        chunked_dict = {}
        idx_b = idx_chunk * chunk_size
        idx_e = idx_b + chunk_size
        if idx_e >= size_dict: idx_e = size_dict

        for _ in range(idx_e - idx_b):
            k, v = next(kv_iter)
            chunked_dict[k] = v
        chunked_dict_in_list.append(chunked_dict)

        if idx_e == size_dict: break

    return chunked_dict_in_list




class ConfusionMatrix:
    def __init__(self, res_dict):
        self.res_dict = res_dict


    def reduce_confusion(self, label):
        ''' Given a label, reduce multiclass confusion matrix to binary
            confusion matrix.
        '''
        res_dict    = self.res_dict
        labels      = res_dict.keys()
        labels_rest = [ i for i in labels if not i == label ]

        # Early return if non-exist label is passed in...
        if not label in labels: 
            print(f"label {label} doesn't exist!!!")
            return None

        # Obtain true positive...
        tp = len(res_dict[label][label])
        fp = sum( [ len(res_dict[label][i]) for i in labels_rest ] )
        tn = sum( sum( len(res_dict[i][j]) for j in labels_rest ) for i in labels_rest )
        fn = sum( [ len(res_dict[i][label]) for i in labels_rest ] )

        return tp, fp, tn, fn


    def get_metrics(self, label):
        # Early return if non-exist label is passed in...
        confusion = self.reduce_confusion(label)
        if confusion is None: return None

        # Calculate metrics...
        tp, fp, tn, fn = confusion
        accuracy    = (tp + tn) / (tp + tn + fp + fn) if tp + tn + fp + fn > 0           else None
        precision   = tp / (tp + fp)                  if tp + fp > 0                     else None
        recall      = tp / (tp + fn)                  if tp + fn > 0                     else None
        specificity = tn / (tn + fp)                  if tn + fp > 0                     else None
        f1_inv      = (1 / precision + 1 / recall)    if tp > 0                          else None
        f1          = 2 / f1_inv                      if f1_inv is not None              else None

        return accuracy, precision, recall, specificity, f1




class TorchModelAttributeParser:

    def __init__(self): return None

    def reset_attr_dict(self):

        module_to_attr_dict = {
            nn.Conv2d : {
                "in_channels"  : None,
                "out_channels" : None,
                "kernel_size"  : None,
                "stride"       : None,
                "padding"      : None,
                "dilation"     : None,
                "groups"       : None,
                "bias"         : None,
                "padding_mode" : None,
                "device"       : None,
                "dtype"        : None,
            },

            nn.BatchNorm2d : {
                "num_features"        : None,
                "eps"                 : None,
                "momentum"            : None,
                "affine"              : None,
                "track_running_stats" : None,
                "device"              : None,
                "dtype"               : None,
            },

            nn.MaxPool2d : {
                "kernel_size"    : None,
                "stride"         : None,
                "padding"        : None,
                "dilation"       : None,
                "return_indices" : None,
                "ceil_mode"      : None,
            },
        }

        return module_to_attr_dict


    def parse(self, model):
        module_to_attr_dict = self.reset_attr_dict()

        model_type = type(model)
        attr_dict  = module_to_attr_dict.get(model_type, {})

        for attr in attr_dict.keys():
            attr_dict[attr] = getattr(model, attr, None)

        return model_type, attr_dict




class NNSize:
    """ Derive the output size of a conv net. """

    def __init__(self, size_y, size_x, channels, conv_dict):
        self.size_y      = size_y
        self.size_x      = size_x
        self.channels    = channels
        self.conv_dict   = conv_dict
        self.method_dict = { nn.Conv2d    : self.get_shape_from_conv2d,
                             nn.MaxPool2d : self.get_shape_from_pool }

        logger.info("___/ NEURAL NETWORK SHAPE \___")

        return None


    def shape(self):
        for layer_name, (model_type, model_attr_tuple) in self.conv_dict.items():
            if model_type not in self.method_dict: continue

            logger.info(f"layer: {layer_name}, {model_type}")
            logger.info(f"in : {self.channels}, {self.size_y}, {self.size_x}")

            #  Obtain the size of the new volume...
            self.channels, self.size_y, self.size_x = \
                self.method_dict[model_type](**model_attr_tuple)

            logger.info(f"out: {self.channels}, {self.size_y}, {self.size_x}")
            logger.info(f"")

        return self.channels, self.size_y, self.size_x


    def get_shape_from_conv2d(self, **kwargs):
        """ Returns the dimension of the output volumne. """
        size_y       = self.size_y
        size_x       = self.size_x
        out_channels = kwargs["out_channels"]
        kernel_size  = kwargs["kernel_size"]
        stride       = kwargs["stride"]
        padding      = kwargs["padding"]

        kernel_size = kernel_size[0] if isinstance(kernel_size, tuple) else kernel_size
        stride      = stride[0]      if isinstance(stride     , tuple) else stride
        padding     = padding[0]     if isinstance(padding    , tuple) else padding

        out_size_y = (size_y - kernel_size + 2 * padding) // stride + 1
        out_size_x = (size_x - kernel_size + 2 * padding) // stride + 1

        return out_channels, out_size_y, out_size_x


    def get_shape_from_pool(self, **kwargs):
        """ Return the dimension of the output volumen. """
        size_y       = self.size_y
        size_x       = self.size_x
        out_channels = self.channels
        kernel_size  = kwargs["kernel_size"]
        stride       = kwargs["stride"]
        padding      = kwargs["padding"]
        dilation     = kwargs["dilation"]
        ceil_mode    = kwargs["ceil_mode"]

        if isinstance(kernel_size, (int, float, str, bool)):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, (int, float, str, bool)):
            stride = (stride, stride)
        if isinstance(padding, (int, float, str, bool)):
            padding = (padding, padding)
        if isinstance(dilation, (int, float, str, bool)):
            dilation = (dilation, dilation)

        out_size_y = (size_y + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1
        out_size_x = (size_x + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[1] + 1

        out_size_y = ceil(out_size_y) if ceil_mode else floor(out_size_y)
        out_size_x = ceil(out_size_x) if ceil_mode else floor(out_size_x)

        return out_channels, out_size_y, out_size_x




class Config:
    def __init__(self, name, **kwargs):
        logger.info(f"___/ Configure {name} \___")

        # Set values of attributes that are not known when obj is created
        for k, v in kwargs.items():
            setattr(self, k, v)
            logger.info(f"KV - {k:16} : {v}")




def save_checkpoint(model, optimizer, scheduler, epoch, loss_min, path):
    torch.save({
        'epoch'               : epoch,
        'loss_min'            : loss_min,
        'model_state_dict'    : model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }, path)




def load_checkpoint(model, optimizer, scheduler, path):
    checkpoint = torch.load(path)
    if model     is not None: model.module.load_state_dict(checkpoint['model_state_dict']) \
                              if hasattr(model, 'module') else \
                              model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None: optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler is not None: scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    return checkpoint['epoch'], checkpoint['loss_min']




def remove_module_from_state_dict(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        new_key = key.replace("module.", "")  # Remove the "module." prefix
        new_state_dict[new_key] = value
    return new_state_dict




def init_weights(module):
    # Initialize conv2d with Kaiming method...
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight.data, nonlinearity = 'relu')

        # Set bias zero since batch norm is used...
        module.bias.data.zero_()




def print_layers(module, max_depth=1, current_indent_width=0, prints_module_name=True):
    """
    Recursively prints the layers of a PyTorch module.  (Keep printing child
    element with a depth first search approach.)

    Args:
    - module (nn.Module): The current module or layer to print.
    - current_indent_width (int): The current level of indentation for printing.
    - prints_name (bool): Flag to determine if the name of the module should be printed.
    """

    def _print_current_layer(module, depth=0, current_indent_width=0, prints_module_name=True):
        # Define a prefix based on current indent level
        prefix = '  ' * current_indent_width

        # Print the name and type of the current module
        if prints_module_name: print(f"{module.__class__.__name__}", end = "")
        print()

        # Check if the current module has children
        # If it does, recursively print each child with an increased indentation level
        if depth < max_depth and list(module.children()):
            for name, child in module.named_children():
                print(f"{prefix}- ({name}): ", end = "")
                _print_current_layer(child, depth + 1, current_indent_width + 1, prints_module_name)

    _print_current_layer(module, current_indent_width=current_indent_width, prints_module_name=prints_module_name)
