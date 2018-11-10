import GenericUtilities as gu
import HistogramUtilities as hu
import Filtering as filter
import numpy as np
import TestingFunction as test
import cv2


#ENVIROMENTAL VARIABLES
BASE_IMAGES_PATH = "/Users/javier/Documents/VA_PRACTICAS/Images"


def test_high_boost():
    # LOAD IMAGE IN GRAYSCALE MODE
    grayscale_image = gu.load_image(BASE_IMAGES_PATH + "/LenaRGB.jpg", gu.GRAY)
    if grayscale_image is not None:
        filtered_image = filter.high_boost(grayscale_image, 0.5, 'gaussian', 1.5)
        imageList = []
        imageList.append(filtered_image)
        imageList.append(grayscale_image)
        imageList.append(grayscale_image)
        gu.multiplot(imageList)


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
        adjusted_img = hu.adjust_intensity(grayscale_image, [], [0.023529411764705882, 0.9176470588235294])
        gu.print_histogram(grayscale_image, adjusted_img)
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
    a = a/255

    # out = np.array([[69, 95, 116, 125, 129, 132, 68, 92, 110, 120, 126, 132, 66, 86
    #                 , 104, 114, 124, 132, 62, 78, 94, 108, 120, 129, 57, 69, 83, 98,
    #                112, 124, 53, 60, 71, 85, 100, 114]]).reshape((6, 6))
    # out = cv2.normalize(out, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    kernel = np.array([[0.1, 0.1, 0.1],
                       [0.1, 0.2, 0.1],
                       [0.1, 0.1, 0.1]])

    #output = filter.filter_image(a, kernel)
    output = filter.new_convolve(a, kernel)
    gu.image_plot(output)


def test_gauss_kernel1D():
    sigma = 30
    kernel = filter.gaussKernel1D(sigma)
    filter.plot_gaussian(kernel)

def dilate():
    kernel = np.array([1., 1.])
    img = np.array([[1., 0., 0., 0.],
                    [1., 0., 0., 0.],
                    [0., 1., 1., 0.],
                    [0., 1., 0., 0.],
                    [0., 1., 0., 0.]])
    # Find the image sizes
    filas, columnas = img.shape
    convolved_img = filter.new_convolve(img, kernel)

    for i in range(0, filas):
        for j in range(0, columnas):
            convolved_img[i][j] = max(min(convolved_img[i][j], 1), 0)

    return convolved_img


if __name__ == "__main__":

    print("RUNNING P1 MAIN")

    dilate()
    # test_adjust_intensisty()
    # test_convolution()
    # test_gauss_kernel1D()
    # test_gaussian_filter()
    # test_median_filter()
    # test_high_boost()
