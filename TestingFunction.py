import scipy.ndimage as nd
import scipy.signal as sg

def gaussianFilter(image, sigma):
    return nd.gaussian_filter(image, sigma)


def gaussian1D(sigma):
    return nd.gaussian_filter1d(input, sigma)


def medianFilter(image, filterSize=None):
    array = np.arange(1,5,1)
    array.cpm
    return sg.medfilt(image, filterSize)
