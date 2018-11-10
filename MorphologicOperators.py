import numpy as np
import GenericUtilities as gu
import Filtering as filter
import scipy.ndimage.morphology as nd

#ENVIROMENTAL VARIABLES
BASE_IMAGES_PATH = "/Users/javier/Documents/VA_PRACTICAS/Images"


def dilate(img, kernel):

    # Find the image sizes
    filas, columnas = img.shape
    convolved_img = filter.filter_image(img, kernel)

    # Check the result and adapt to only 1 or 0 output values
    for i in range(0, filas):
        for j in range(0, columnas):
            convolved_img[i][j] = max(min(convolved_img[i][j], 1), 0)

    return convolved_img


def erode(img, kernel):

    # Find the image sizes
    filas, columnas = img.shape
    convolved_img = filter.filter_image(img, kernel)
    # Kernel sum of ones for after check
    kernsum = kernel.sum()

    # Check the result. If pixel value == kernel.sum() -> All ones match, draw a 1
    # Check the result and adapt to only 1 or 0 output values
    for i in range(0, filas):
        for j in range(0, columnas):
            if convolved_img[i, j] == kernsum:
                convolved_img[i][j] = 1
            else:
                convolved_img[i][j] = 0
    return convolved_img


def opening(image, kernel):
    # Erode followed of a dilate
    output_img = erode(image, kernel)
    output_img = dilate(output_img, kernel)
    return output_img


def closing(image,kernel):
    # Dilate followed by erode
    output_img = dilate(image, kernel)
    output_img = erode(output_img, kernel)
    return output_img


if __name__ == "__main__":

    print("RUNNING P1 MAIN")
    # grayscale_image = gu.load_image(BASE_IMAGES_PATH+"/erode.png", gu.GRAY)

    image = np.array([[1., 0., 0., 0.],
                      [1., 0., 0., 0.],
                      [0., 1., 1., 0.],
                      [0., 1., 0., 0.],
                      [0., 1., 0., 0.]])
    kern = np.array([[1., 1.]])
    # kern2 = np.ones((15, 15))
    # out = erode(grayscale_image, kern2)
    out = dilate(image, kern)
    gu.image_plot(out)
    # Test of dilate
    # dilate(image,kern)




