#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging


def main():
    #setup a logger
    logger = logging.getLogger('AtomicSensorSimulation')
    logger.setLevel(logging.DEBUG)
    file_handler = logging.FileHandler('spam.log')
    file_handler.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.ERROR)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return 0


if __name__ == "__main__":
    main()