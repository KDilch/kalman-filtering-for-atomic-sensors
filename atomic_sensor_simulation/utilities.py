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


def plot_data(xs, ys, data_labels=None, title="", output=None, **kwargs):
    """
    :param xs: np.array with arrays containing X data
    :param ys: np.array with arrays containing Y data
    :param data_labels: a list containing labels for plotted sets of data
    :param title: a string containing a plot title
    :param output: a string containing an output path (optional)
    :param kwargs: is_show [bool, default: False],
                   is_legend [bool, default: False],
    :return: (void)
    """
    logger = logging.getLogger(__name__)
    logger.info('Plotting data initiated.')
    is_show, is_legend = kwargs.get('is_legend', False), kwargs.get('is_show', False)
    fig = plt.figure()
    fig.patch.set_facecolor('white')
    ax = fig.add_subplot(111)
    for index, x in enumerate(xs):
        ax.plot(x, ys[index], label=data_labels[index])
    ax.grid(True)
    ax.set_title(title)
    if is_legend:
        plt.legend()
    if is_show:
        plt.show()
        plt.close()
    if output:
        dirpath = os.path.dirname(output)
        if not os.path.exists(dirpath):
            if dirpath != "":
                os.makedirs(dirpath)
            else:
                logger.info('Output path for a plot not specified. Plot wasn\'t be saved.')
                return
        plt.savefig(output, format="svg")
    return


def generate_data_arr_for_plotting(all_xs, all_ys, labels, bools):
    selected_xs = []
    selected_ys = []
    selected_labels = []
    for index, bool in enumerate(bools):
        if bool:
            selected_xs.append(all_xs[index])
            selected_ys.append(all_ys[index])
            selected_labels.append(labels[index])
    return selected_xs, selected_ys, selected_labels


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
