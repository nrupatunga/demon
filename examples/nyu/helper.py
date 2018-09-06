"""
File: helper.py
Author: Nrupatunga
Email: nrupatunga@whodat.in
Github: https://github.com/nrupatunga
Description: Some of helper functions
"""


def indices(a, func):
    """Implementaion of find() like function in matlab

    :a: TODO
    :func: TODO
    :returns: TODO

    """
    return [i for (i, val) in enumerate(a) if func(val)]
