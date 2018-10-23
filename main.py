import GenericUtilities as gu
import HistogramUtilities as hu
import Filtering as filter
import numpy as np
import TestingFunction as test

#ENVIROMENTAL VARIABLES
BASE_IMAGES_PATH = "/Users/javier/Documents/VA_PRACTICAS/Images"


def test_gaussian_filter():
    # LOAD IMAGE IN GRAYSCALE MODE
    grayscale_image = gu.load_image(BASE_IMAGES_PATH + "/LenaRGB.jpg", gu.GRAY)
    if grayscale_image is not None:
        sigma = 30
        filtered_image = filter.gaussianFilter(grayscale_image, sigma)
        test_image = test.gaussianFilter(grayscale_image,sigma)
        imageList = []
        imageList.append(filtered_image)
        imageList.append(grayscale_image)
        imageList.append(test_image)
        gu.multiplot(imageList)


def test_median_filter():
    grayscale_image = gu.load_image(BASE_IMAGES_PATH + "/LenaRGB.jpg", gu.GRAY)
    kernel_size = 7
    img = filter.median_filter(grayscale_image, kernel_size)
    gu.image_plot(img)


def test_adjust_intensisty():

    # LOAD IMAGE IN GRAYSCALE MODE
    grayscale_image = gu.load_image(BASE_IMAGES_PATH+"/LenaRGB.jpg", gu.GRAY)
    if grayscale_image is not None:
        # 3.1 Alteracion del rango dinamico
        hu.adjust_intensity(grayscale_image, [], [0, 255])
        # 3.2 Ecualizacion de histograma


def test_convolution():
    a = np.array([[45, 60, 98, 127, 132, 133, 137, 133],
              [46, 65, 98, 123, 126, 128, 131, 133],
              [47, 65, 96, 115, 119, 123, 135, 137],
              [47, 63, 91, 107, 113, 122, 138, 134],
              [50, 59, 80, 97, 110, 123, 133, 134],
              [49, 53, 68, 83, 97, 113, 128, 133],
              [50, 50, 58, 70, 84, 102, 116, 126],
              [50, 50, 52, 58, 69, 86, 101, 120]])

    kernel = np.array([[0.1, 0.1, 0.1],
                   [0.1, 0.2, 0.1],
                   [0.1, 0.1, 0.1]])

    output = filter.filter_image(a, kernel)


def test_gauss_kernel1D():
    sigma = 30
    kernel = filter.gaussKernel1D(sigma)
    filter.plot_gaussian(kernel)


if __name__ == "__main__":

    print("RUNNING P1 MAIN")
    # test_convolution()
    # test_gauss_kernel1D()
    # test_gaussian_filter()
    test_median_filter()

