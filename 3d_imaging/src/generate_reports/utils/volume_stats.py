"""
Contains various functions for computing statistics over 3D volumes
"""
import numpy as np

def Dice3d(a, b):
    """
    This will compute the Dice Similarity coefficient for two 3-dimensional volumes
    Volumes are expected to be of the same size. We are expecting binary masks -
    0's are treated as background and anything else is counted as data

    Arguments:
        a {Numpy array} -- 3D array with first volume
        b {Numpy array} -- 3D array with second volume

    Returns:
        float
    """
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    ba = (a > 0).astype(int)
    bb = (b > 0).astype(int)
    n = (ba + bb == 2).sum()
    magna = ba.sum()
    magnb = bb.sum()

    return ((2 * n) / (magna + magnb))

def Jaccard3d(a, b):
    """
    This will compute the Jaccard Similarity coefficient for two 3-dimensional volumes
    Volumes are expected to be of the same size. We are expecting binary masks - 
    0's are treated as background and anything else is counted as data

    Arguments:
        a {Numpy array} -- 3D array with first volume
        b {Numpy array} -- 3D array with second volume

    Returns:
        float
    """
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    ba = (a > 0).astype(int)
    bb = (b > 0).astype(int)
    n = (ba + bb == 2).sum()
    magna = ba.sum()
    magnb = bb.sum()

    return (n / (magna + magnb - n))


def Sensitivity3D(a, b):
    """
    Returns the sensitivity (true positive rate) between the predictions a and labels b
    :param a: {numpy array} 3D volume
    :param b: {numpy array} 3D volume
    :return:
    """

    ba = (a > 0).astype(int)
    bb = (b > 0).astype(int)

    true_positives  = (ba + bb == 2).sum()
    total_positives = bb.sum()

    return true_positives / total_positives


def Specificity3D(a, b):
    """
    Returns the specificity (true negative rate) between the predictions a and labels b
    :param a: {numpy array} 3D volume
    :param b: {numpy array} 3D volume
    :return:
    """

    ba = (a > 0).astype(int)
    bb = (b > 0).astype(int)

    true_negatives  = (ba + bb == 0).sum()
    total_negatives = (bb ==  0).sum()

    return true_negatives / total_negatives
