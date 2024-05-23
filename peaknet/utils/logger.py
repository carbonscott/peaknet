#!/usr/bin/env python
# -*- coding: utf-8 -*-

import logging
import os

from datetime import datetime

logger = logging.getLogger(__name__)

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
