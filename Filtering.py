from skimage.exposure import rescale_intensity
import matplotlib.rcsetup as rcsetup
import matplotlib.pyplot as plt
import numpy as np
import cv2


def filter_image(image, kernel):
    # Calculamos las dimensione de la imagen y del kernel
    (imageH, imageW) = image.shape[:2]
    (kernelH, kernelW) = kernel.shape[:2]

    # Añadimos paddig de ceros en los bordes de la imagen
    # para poder aplicar el filtro sin reducir el tamaño original de esta
    # TODO si el filtro no es siempre cuadrado, cambiar el paddig para que haya padHorizontal padVertical
    pad = (kernelW - 1) // 2
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
                               cv2.BORDER_REPLICATE)
    output = np.zeros((imageH, imageW), dtype="float32")

    # Iteramos la imagen
    for y in np.arange(pad,imageH+pad): # Empezamos en el indice despues del padding, has la ultima casilla !=0
        for x in np.arange(pad, imageW + pad):
            # Recortamos la sumbatriz de la imagen que solapa con el kernel
            window = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
            # Multiplicamos elemento por elemento la ventana de la imagen por el kernel
            # Luego sumamos todos los resultados
            k = (window * kernel).sum()
            # Guardamos la suma en la posicion correspondiente de la matriz original (sin el padding)
            output[y - pad, x - pad] = k

    # Durante la convolucion se realizan multiplicaciones que pueden dar resultados >255
    # para solucionarlo, re-escalamos la imagen dentro del rango
    # TODO consultar si la funcion de reescalado se puede usar, o se implementa
    output = rescale_intensity(output, in_range=(0, 255))
    # Convertimos los resultados en decimal a uint8
    output = (output * 255).astype("uint8")

    # return the output image
    return output


def gaussian_distribution(x, sigma):
    exponent = -pow(x, 2) / 2 / pow(sigma, 2)
    fraction = 1 / (np.sqrt(2*np.pi) * sigma)
    return fraction * pow(np.e, exponent)


def plot_gaussian(values_array):
    x_axis = np.arange(0, len(values_array[0]), 1)
    plt.interactive(False)
    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.plot(x_axis, values_array[0])
    plt.show()
    print("HELLO WAITINH")



def gaussKernel1D(sigma):

    N = int(2 * np.floor(3*sigma) + 1)
    kernel = np.zeros((1,N))
    middle_index = int(np.floor((N/2) ))
    for x in np.arange(0, N):
        gaussian_index = (x-middle_index)
        kernel[0][x] = gaussian_distribution(gaussian_index, sigma)
    return kernel


def gaussianFilter(inImage, sigma):

    kernel1N = gaussKernel1D(sigma)
    kernelN1 = np.transpose([kernel1N])

    outImg = filter_image(inImage, kernel1N)
    return filter_image(outImg, kernelN1)
