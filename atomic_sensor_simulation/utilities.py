#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import logging.config
import matplotlib.pyplot as plt
import logging
from scipy.linalg import expm


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


class operable:
    def __init__(self, f):
        self.f = f

    def __call__(self, x):
        return self.f(x)

    def __mul__(self, other):
        return operable(lambda x: self(x) * other(x))

    def __add__(self, other):
        return operable(lambda x: self(x) + other(x))


def create_matrix_of_functions(matrix):
    def matrix_of_functions_obj(x):
        matrix_flat = matrix.flatten()
        shape = np.shape(matrix)
        evaluated_matrix = np.empty_like(matrix_flat)
        for index, element in np.ndenumerate(matrix_flat):
            evaluated_matrix[index] = matrix_flat[index](x)
        return np.reshape(evaluated_matrix, shape)
    return matrix_of_functions_obj


def exp_matrix_of_functions(matrix):
    return lambda time: expm(matrix(time))




