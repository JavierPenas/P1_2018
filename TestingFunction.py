import scipy.ndimage as nd
import scipy.signal as sg
import numpy as np

def gaussianFilter(image, sigma):
    return nd.gaussian_filter(image, sigma)


def gaussian1D(sigma):
    return nd.gaussian_filter1d(input, sigma)


def medianFilter(image, filterSize=None):

    return sg.medfilt(image, filterSize)

