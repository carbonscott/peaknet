#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

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


def is_action_due(iter_num, no_action_interval = None):
    """ iter_num is zero-based.
    """
    return no_action_interval is not None and (iter_num + 1) % no_action_interval == 0
