#!/usr/bin/env python
# -*- coding: utf-8 -*-
import logging
from logging.handlers import RotatingFileHandler
import os


class Logger(object):

    def __init__(self, name, log_file_path=None):
        self.logger = logging.getLogger(name)
        self.__log_file_path = log_file_path
        self.log_file_handler = self.__create_log_file_handler()
        self.console_handler = logging.StreamHandler()
        self.__setup_logger()

    @property
    def log_file_path(self):
        return self.__log_file_path

    @property
    def log_dir_path(self):
        return os.path.dirname(self.__log_file_path)

    def __create_log_file_handler(self):
        if self.__log_file_path is not None:
            if os.path.exists(self.log_dir_path):
                return RotatingFileHandler(self.__log_file_path,
                                           mode='a',
                                           maxBytes=5*1024*1024,
                                           backupCount=2,
                                           encoding=None,
                                           delay=0)
            else:
                os.makedirs(self.log_dir_path)
                return logging.FileHandler(self.__log_file_path)
        else:
            return None

    def __setup_logger(self):
        self.logger.setLevel(logging.DEBUG)
        self.log_file_handler.setLevel(logging.DEBUG)
        self.console_handler.setLevel(logging.ERROR)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        self.log_file_handler.setFormatter(formatter)
        self.console_handler.setFormatter(formatter)
        self.logger.addHandler(self.log_file_handler)
        self.logger.addHandler(self.console_handler)
