from skimage.exposure import rescale_intensity
import matplotlib.pyplot as plt
import numpy as np
import cv2



def high_boost(image, A, method, param):

    # If method or param is None, we can't apply any filter
    if (method is None) or (param is None):
        return image
    # Calculate the filtered image
    if method == 'gaussian':
        filtered_image = gaussianFilter(image, param)
    else:
        if method == 'median':
            filtered_image = median_filter(image, param)
        else:
            # In case the specified filtering method is not valid
            return None

    if A >= 0:
        # If A is a positive value, we preserve part of original image info
        image = np.multiply(image, A).astype(np.uint8)
        return np.subtract(image, filtered_image)
    else:
        # In other case, we only apply filter
        return filtered_image


def median_filter(image, kernel_size):
    #Calculamos la matriz del kernel
    kernel = np.repeat(kernel_size, image.ndim)
    kernel = np.ones(kernel)
    #Calculamos imagen de salida
    (imageH, imageW) = image.shape[:2]
    output = np.zeros((imageH, imageW), dtype="float32")
    pad = (kernel_size - 1) // 2
    # TODO PREGUNTAR SI MEJOR REPLICAR BORDE O RELLENAR CON 0 | 1 | NO usarlos
    image = cv2.copyMakeBorder(image, pad, pad, pad, pad,
                               cv2.BORDER_REPLICATE)
    # Iteramos la imagen
    for y in np.arange(pad, imageH+pad): # Empezamos en el indice despues del padding, has la ultima casilla !=0
        for x in np.arange(pad, imageW + pad):
            # Recortamos la sumbatriz de la imagen que solapa con el kernel
            window = image[y - pad:y + pad + 1, x - pad:x + pad + 1]
            # Multiplicamos elemento por elemento la ventana de la imagen por el kernel
            # Luego sumamos todos los resultados
            k = np.median((window * kernel))
            # Guardamos la suma en la posicion correspondiente de la matriz original (sin el padding)
            output[y - pad, x - pad] = int(k)
    return output.astype(np.uint8)


def filter_image(image, kernel):

    #Obtenemos dimensiones
    (filas, columnas) = image.shape
    try:
        (k_filas, k_columnas) = kernel.shape
    except ValueError:
        (k_filas, k_columnas) = (1, len(kernel))

    kern_mid_col = k_columnas // 2
    kern_mid_row = k_filas // 2

    # Creacion imagen para operar
    convolving_img = cv2.copyMakeBorder(image, kern_mid_row, kern_mid_row, kern_mid_col, kern_mid_col,
                               cv2.BORDER_CONSTANT, value=0)
    # Creacion imagen salida
    output_image = np.zeros(image.shape)

    # Iteracion y convolucion
    for i in range(0, filas):
        for j in range(0, columnas):
            # Window-sized kernel calculation
            vert_size = i+k_filas
            horz_size = j+k_columnas
            window = convolving_img[i:vert_size, j:horz_size]
            # Calculo de la convolucion
            k = np.sum(window * kernel)
            output_image[i, j] = k

    return output_image


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
    kernel = []
    middle_index = int(np.floor((N/2) ))
    for x in np.arange(0, N):
        gaussian_index = (x-middle_index)
        kernel.append(gaussian_distribution(gaussian_index, sigma))
    return np.asarray(kernel)


def gaussianFilter(inImage, sigma):
    kernel1N = gaussKernel1D(sigma)
    kernelN1 = np.transpose([kernel1N])

    outImg = filter_image(inImage, kernel1N)
    return filter_image(outImg, kernelN1)
