#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from time import sleep


def listener_process(queue):
    while True:
        while not queue.empty():
            record = queue.get()
            logger = logging.getLogger(record.name)
            logger.handle(record)
        sleep(1)