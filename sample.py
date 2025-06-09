import numpy as np


def sample_normal(shape, mean=0, std=1):
    """
    Sample a desired amount of parameters from a Gaussian distribution with mean = 0 and var = 1
    """
    return np.random.normal(size=shape, loc=mean, scale=std)
