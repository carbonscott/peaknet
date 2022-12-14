#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import logging
import torch
from peaknet.datasets.images import ConfigDataset, MiniSFXDataset
from peaknet.methods.unet    import UNet
from peaknet.model           import ConfigPeakFinderModel, PeakFinderModel
from peaknet.trainer         import ConfigTrainer, Trainer
from peaknet.validator       import ConfigValidator, LossValidator
from peaknet.utils           import EpochManager, MetaLog
from datetime import datetime
## from image_preprocess import DatasetPreprocess
## from image_no_reg_preprocess import DatasetPreprocess
import socket

# Set up parameters for an experiment...
fl_csv                = 'datasets.csv'
drc_project           = os.getcwd()
size_sample_train     = 40
size_sample_validate  = 40
frac_train            = 0.005
frac_validate         = None
dataset_usage         = 'train'

size_batch     = 2
lr             = 1e-3
seed           = 0

# Clarify the purpose of this experiment...
hostname = socket.gethostname()
comments = f"""
            Hostname: {hostname}.

            Online training.

            Sample size (train)     : {size_sample_train}
            Sample size (validate)  : {size_sample_validate}
            Batch  size             : {size_batch}
            lr                      : {lr}

            """

# [[[ LOGGING ]]]
# Create a timestamp to name the log file...
now = datetime.now()
timestamp = now.strftime("%Y_%m%d_%H%M_%S")

# Configure the location to run the job...
drc_cwd = drc_project

# Set up the log file...
fl_log         = f"{timestamp}.train.log"
DRCLOG         = "logs"
prefixpath_log = os.path.join(drc_cwd, DRCLOG)
if not os.path.exists(prefixpath_log): os.makedirs(prefixpath_log)
path_log = os.path.join(prefixpath_log, fl_log)

# Config logging behaviors
logging.basicConfig( filename = path_log,
                     filemode = 'w',
                     format="%(asctime)s %(levelname)s %(name)-35s - %(message)s",
                     datefmt="%m/%d/%Y %H:%M:%S",
                     level=logging.INFO, )
logger = logging.getLogger(__name__)

# Create a metalog to the log file, explaining the purpose of this run...
metalog = MetaLog( comments = comments )
metalog.report()

# [[[ DATASET ]]]
# Config the dataset...
config_dataset = ConfigDataset( fl_csv                = fl_csv,
                                drc_project           = drc_project,
                                size_sample           = size_sample_train, 
                                dataset_usage         = dataset_usage,
                                trans                 = None,
                                frac_train            = frac_train,
                                frac_validate         = frac_validate,
                                seed                  = seed, )

# Define the training set
dataset_train = MiniSFXDataset(config_dataset)
dataset_train.cache_img(dataset_train.miniset)
## dataset_train.report()

# Report training set...
config_dataset.report()

# Define validation set...
config_dataset.size_sample   = size_sample_validate
config_dataset.dataset_usage = 'validate'
config_dataset.report()
dataset_validate = MiniSFXDataset(config_dataset)
dataset_validate.cache_img(dataset_validate.miniset)


# [[[ IMAGE ENCODER ]]]
# Config the encoder...
img = dataset_train[0][0]
size_y, size_x = img.shape[-2:]
method = UNet(in_channels = 1, out_channels = 1)


# [[[ MODEL ]]]
# Config the model...
config_peakfinder = ConfigPeakFinderModel( method = method )
model = PeakFinderModel(config_peakfinder)

# Initialize weights...
def init_weights(module):
    if isinstance(module, (torch.nn.Embedding, torch.nn.Linear)):
        module.weight.data.normal_(mean = 0.0, std = 0.02)
model.apply(init_weights)


# [[[ CHECKPOINT ]]]
DRCCHKPT         = "chkpts"
prefixpath_chkpt = os.path.join(drc_cwd, DRCCHKPT)
if not os.path.exists(prefixpath_chkpt): os.makedirs(prefixpath_chkpt)
fl_chkpt         = f"{timestamp}.train.chkpt"
path_chkpt       = os.path.join(prefixpath_chkpt, fl_chkpt)


# [[[ TRAINER ]]]
# Config the trainer...
config_train = ConfigTrainer( path_chkpt     = path_chkpt,
                              num_workers    = 1,
                              batch_size     = size_batch,
                              pin_memory     = True,
                              shuffle        = False,
                              lr             = lr, )

# Training...
trainer = Trainer(model, dataset_train, config_train)


# [[[ VALIDATOR ]]]
config_validator = ConfigValidator( path_chkpt     = None,
                                    num_workers    = 1,
                                    batch_size     = size_batch,
                                    pin_memory     = True,
                                    shuffle        = False,
                                    lr             = lr,)
validator = LossValidator(model, dataset_validate, config_validator)


# [[[ EPOCH MANAGER ]]]
max_epochs = 360
epoch_manager = EpochManager(trainer = trainer, validator = validator, max_epochs = max_epochs)
epoch_manager.run()
