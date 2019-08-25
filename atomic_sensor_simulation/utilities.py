#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import json
import numpy as np
import logging.config
import matplotlib.pyplot as plt
import logging
from scipy.linalg import expm
from scipy.integrate import quad
from pathlib import Path
from importlib import import_module


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


def import_config_from_path(module_name):
    module_object = import_module(module_name)
    return getattr(module_object, 'config')


def load_logging_config(default_path='logging.json', default_level=logging.INFO):
    """Setup logging configuration
    """
    path = default_path
    if os.path.exists(path):
        with open(path, 'rt') as f:
            config = json.load(f)
        info_file_path = os.path.dirname(config['handlers']['info_file_handler']['filename'])
        error_file_path = os.path.dirname(config['handlers']['error_file_handler']['filename'])
        if not os.path.exists(info_file_path):
            os.makedirs(info_file_path)
        if not os.path.exists(error_file_path):
            os.makedirs(error_file_path)
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


def eval_matrix_of_functions(matrix, x):
    matrix_flat = matrix.flatten()
    shape = np.shape(matrix)
    evaluated_matrix = np.empty_like(matrix_flat)
    for index, element in np.ndenumerate(matrix_flat):
        evaluated_matrix[index] = matrix_flat[index](x)
    return np.reshape(evaluated_matrix, shape)


def exp_matrix_of_functions(matrix):
    return lambda time: expm(matrix(time))


def integrate_matrix_of_functions(matrix, from_x, to_x):
    matrix_flat = matrix.flatten()
    shape = np.shape(matrix)
    integrated_matrix = np.empty_like(matrix_flat)
    for index, element in np.ndenumerate(matrix_flat):
        integrated_matrix[index] = quad(matrix_flat[index], from_x, to_x)[0]
    return np.reshape(integrated_matrix, shape)


def calculate_error(W, x, x_est):
    x = np.array([x]).T
    Sigma = np.dot((x-x_est), (x-x_est).T)
    return np.trace(np.dot(W, Sigma))

def compute_squred_error_from_covariance(Sigma, index):
    return Sigma[index][index]
