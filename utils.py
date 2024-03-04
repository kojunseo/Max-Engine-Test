import numpy as np
import torch


def accuracy(output, labels):
    """
    output = [0, 2, 3, 1]
    labels = [0, 2, 2, 1]

    get the number of correct predictions
    """
    return (output == labels).sum().item()