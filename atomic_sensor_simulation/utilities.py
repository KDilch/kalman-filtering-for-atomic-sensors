#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import logging.config
import matplotlib.pyplot as plt
import logging


def stringify_namespace(namespace):
    """
    :param argv: namespace
    :return: string
    """
    __str = ''
    for arg in namespace.__dict__:
        if arg:
            __str += " --" + arg + ": " + str(namespace.__dict__[arg])
    return __str


def load_logging_config(default_path='logging.json', default_level=logging.INFO):
    """Setup logging configuration
    """
    path = default_path
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
    return


def plot_data(x, y, **kwargs):
    """
    :param x: np.array with X data
    :param y: np.array with Y data
    :param kwargs: xlabel [str, defaullt: ""],
                   ylabel[str, default: ""],
                   title [str, default: ""],
                   is_show [bool, default: False],
                   output_path [str, default: ""]
    :return:
    """
    logger = logging.getLogger(__name__)
    logger.info('Plotting data initiated.')
    xlabel, ylabel, title, is_show, output = kwargs.get('xlabel',""),\
                                             kwargs.get('ylabel',""),\
                                             kwargs.get('title',""),\
                                             kwargs.get('is_show', False),\
                                             kwargs.get('output', "")
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111)
    ax.plot(x, y)
    ax.grid(True)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if is_show:
        plt.show()
        plt.close()

    dirpath = os.path.dirname(output)
    if not os.path.exists(dirpath):
        if dirpath != "":
            os.makedirs(dirpath)
        else:
            logger.info('Output path for a plot not specified. Plot wasn\'t be saved.')
            return
    plt.savefig(output, format="svg")
    return






