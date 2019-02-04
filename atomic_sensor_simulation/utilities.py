# -*- coding: utf-8 -*-


def stringify_namespace(argv):
    """
    :param argv: namespace
    :return: string
    """
    __str = ''
    for arg in argv.__dict__:
        if arg:
            __str += " --" + arg + ": " + str(argv.__dict__[arg])
        else:
            pass
    return __str
