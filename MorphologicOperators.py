import numpy as np
import GenericUtilities as gu
import Filtering as filter
import cv2

#ENVIROMENTAL VARIABLES
BASE_IMAGES_PATH = "/Users/javier/Documents/VA_PRACTICAS/Images"


def calculate_padding(kernel, center):

    try:
        (k_filas, k_columnas) = kernel.shape
    except ValueError:
        (k_filas, k_columnas) = (1, len(kernel))

    # Si no se me especifica el centro del kernel, lo calculo
    if gu.is_empty_list(center):
        kern_mid_col = k_columnas // 2
        kern_mid_row = k_filas // 2
        return {'DCHA': kern_mid_col, 'IZQ': kern_mid_col, 'ARR': kern_mid_row, 'ABJ': kern_mid_row}
    else:
        kern_mid_col = center[0]
        kern_mid_row = center[1]
        pad_dcha = k_columnas - center[0]
        pad_abajo = k_filas - center[0]
        return {'DCHA': pad_dcha, 'IZQ': kern_mid_col, 'ARR': kern_mid_row, 'ABJ': pad_abajo}


def dilate(img, kernel, center):
    # Find the image sizes
    filas, columnas = img.shape
    try:
        (k_filas, k_columnas) = kernel.shape
    except ValueError:
        (k_filas, k_columnas) = (1, len(kernel))

    padding = calculate_padding(kernel, center)

    # Creacion imagen para operar
    convolving_img = cv2.copyMakeBorder(img, padding.get("ARR"), padding.get("ABJ"), padding.get("IZQ"), padding.get("DCHA"),
                               cv2.BORDER_CONSTANT, value=0)
    dilated_img = cv2.copyMakeBorder(img, padding.get("ARR"), padding.get("ABJ"), padding.get("IZQ"), padding.get("DCHA"),
                               cv2.BORDER_CONSTANT, value=0)
    # Creacion imagen salida
    output_image = np.zeros(image.shape)

    # Iteracion y convolucion
    for i in range(0, filas):
        for j in range(0, columnas):
            # Window-sized kernel calculation
            if img[i, j] == 1:
                vert_size = i+k_filas
                horz_size = j+k_columnas
                window = convolving_img[i:vert_size, j:horz_size]
                # Calculo de la convolucion
                k = np.sum(window * kernel)
                if k > 0:
                    dilated_img[i:vert_size, j:horz_size] = kernel

    return dilated_img[padding.get("ARR"):filas+padding.get("ARR"), padding.get("IZQ"):columnas+padding.get("IZQ"), ]


# def dilate(img, kernel, center):
#
#     # Find the image sizes
#     filas, columnas = img.shape
#     # convolved_img = filter.filter_image(img, kernel)
#     convolved_img = convolve_notCentered(img, kernel, center)
#     # Check the result and adapt to only 1 or 0 output values
#     for i in range(0, filas):
#         for j in range(0, columnas):
#             convolved_img[i][j] = max(min(convolved_img[i][j], 1), 0)
#
#     return convolved_img


def erode(img, kernel, center):

    # Find the image sizes
    filas, columnas = img.shape
    convolved_img = convolve_notCentered(img, kernel, center)
    # convolved_img = filter.filter_image(img, kernel)
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


def convolve_notCentered(image, kernel, center):

    #Obtenemos dimensiones
    (filas, columnas) = image.shape

    try:
        (k_filas, k_columnas) = kernel.shape
    except ValueError:
        (k_filas, k_columnas) = (1, len(kernel))

    padding = calculate_padding(kernel, center)

    # Creacion imagen para operar
    convolving_img = cv2.copyMakeBorder(image, padding.get("ARR"), padding.get("ABJ"), padding.get("IZQ"), padding.get("DCHA"),
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


def opening(image, kernel, center):
    # Erode followed of a dilate
    output_img = erode(image, kernel)
    output_img = dilate(output_img, kernel)
    return output_img


def closing(image,kernel, center):
    # Dilate followed by erode
    output_img = dilate(image, kernel, center)
    output_img = erode(output_img, kernel, center)
    return output_img


def hit_or_miss(inImage, objSEj, bgSE, center):

    bg_image = inImage.copy()
    bg_image[bg_image == 1] = -1
    bg_image[bg_image == 0] = 1
    bg_image[bg_image == -1] = 0

    if not np.equal(np.sum(objSEj+bgSE), np.ones(objSEj.shape).sum()):
        print("Error: elementos estructurantes incoherentes")
    else:
        output1 = erode(inImage, objSEj, center)
        output2 = erode(bg_image, bgSE, center)
        output_img = output1 + output2
        output_img[output_img != 2] = 0
        output_img[output_img == 2] = 1
        return output_img


if __name__ == "__main__":

    print("RUNNING P1 MAIN")
    grayscale_image = gu.load_image(BASE_IMAGES_PATH+"/erode.png", gu.GRAY)

    # image = np.array([[1., 0., 0., 0.],
    #                   [1., 0., 0., 0.],
    #                   [0., 1., 1., 0.],
    #                   [0., 1., 0., 0.],
    #                   [0., 1., 0., 0.]])
    # kern = np.array([[1., 1., 1.], [0., 0., 0.], [0., 0., 0.]])
    # kern = np.array([1., 1.])
    # kern2 = np.ones((25, 25))
    # out = erode(image, kern, [0, 0])
    # out = dilate(image, kern, [0, 0])
    # out = dilate(grayscale_image, kern2, None)
    # gu.image_plot(out)

    #HIT OR MISS
    image = np.array([[0., 0., 0., 0., 0., 0., 0.],
                      [0., 0., 0., 1., 1., 0., 0.],
                      [0., 0., 1., 1., 1., 1., 0.],
                      [0., 0., 1., 1., 1., 1., 0.],
                      [0., 0., 0., 1., 1., 0., 0.],
                      [0., 0., 0., 1., 0., 0., 0.],
                      [0., 0., 0., 0., 0., 0., 0.]])

    kernOne = np.array([[1., 1.],
                        [0., 1.]])

    kernBg = np.array([[0., 0.],
                       [1., 0.]])
    hit_or_miss(image, kernOne, kernBg, [0,0])




