"""
File: helper.py
Author: Nrupatunga
Email: nrupatunga@whodat.in
Github: https://github.com/nrupatunga
Description: Some of helper functions
"""


def indices(a, func):
    """Implementaion of find() like function in matlab

    Args:
        a: input array
        func: input function that needs to be satisfied

    Returns:
        returns the indices satisfying func
    """
    return [i for (i, val) in enumerate(a) if func(val)]
