import numpy as np
import math
import Filtering as filter
import GenericUtilities as gu


def roberts_kernel():
    return np.array([[-1., 0.], [0., 1]]), np.array([[0., -1.], [1., 0.]])


def central_diff_kernel():
    return np.array([[-1, 0, 1]]), np.array([[-1], [0], [1]])


def prewitt_kernel():
    return np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]), np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])


def sobel_kernel():
    return np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]), np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])


def gradient_image(image, operator):
    # Calculamos los kernels segun el metodo seleccionado
    if operator == 'Roberts':
        kern_x, kern_y = roberts_kernel()
    elif operator == 'CentralDiff':
        kern_x, kern_y = central_diff_kernel()
    elif operator == 'Prewitt':
        kern_x, kern_y = prewitt_kernel()
    elif operator == 'Sobel':
        kern_x, kern_y = sobel_kernel()
    else:
        return None

    # Gradiente en X
    gx = filter.filter_image(image, kern_x)
    # Gradiente en Y
    gy = filter.filter_image(image, kern_y)
    # Devolvemos los resultados
    return [gx, gy]


def edge_canny(image, sigma, t_low, t_high):
    # https://es.wikipedia.org/wiki/Algoritmo_de_Canny

    # Comprobamos que el umbral bajo no supere el alto
    # if t_high < t_low :
    #    return None

    # Aplicamos el filtro gaussiano
    image_gaussian = filter.gaussianFilter(image, sigma)

    # Aplicamos un Sobel a la imagen ya suavizada
    [gx, gy] = gradient_image(image_gaussian, 'Sobel')

    # Calculo gradiente del borde y orientacion
    gradiente = np.sqrt((gx ** 2) + (gy ** 2))
    orientacion = np.arctan2(gy, gx)

    # Agrupamos las direcciones de los pixeles en 4 posibles angulos (0, 45,90,135)
    angulos = obtain_angle(orientacion)
    # Eliminamos de matriz gradiente aquellos puntos que no sean maximos
    pixeles_bordes, maximos = noMaxSupression(gradiente, angulos)

    # Realizamos histerisis
    cannyEdges = histerisis(maximos, angulos, pixeles_bordes, t_low, t_high)

    # Obtenemos la matriz de bordes a partir de los puntos
    output_image = np.zeros(image.shape)
    for pos in cannyEdges:
        output_image[pos] = 1

    return output_image


def obtain_angle(orientacion):
    filas, columnas = orientacion.shape

    # r_output = orientacion.copy()
    m_output = orientacion.copy()

    for i in range(0, filas):
        for j in range(0, columnas):
            angle = orientacion[i][j]
            # r_output[i][j] = select_direction(angle)
            m_output[i][j] = select_direction2(angle)

    return m_output


def select_direction2(angle):
    angle = angle * 180 / math.pi
    if angle < 0:
        angle += 360
    # 45 degrees
    if (angle >= 22.5 and angle < 67.5) or (angle >= 202.5 and angle < 247.5):
        return 45
    # 0 degrees
    if (angle >= 337.5 or angle < 22.5) or (angle >= 157.5 and angle < 202.5):
        return 0
    # 45 degrees
    if (angle >= 22.5 and angle < 67.5) or (angle >= 202.5 and angle < 247.5):
        return 45
    # 90 degrees
    if (angle >= 67.5 and angle < 112.5) or (angle >= 247.5 and angle < 292.5):
        return 90
    # 135 degrees
    if (angle >= 112.5 and angle < 157.5) or (angle >= 292.5 and angle < 337.5):
        return 135


def get_perpendicular_neighbours(values, orientacion, posicion_local):
    i, j = posicion_local
    max_filas, max_columnas = values.shape
    n1, n2 = None, None
    if orientacion == 0:
        if i < max_filas - 1:
            n1 = (i + 1, j)
        if i > 0:
            n2 = (i - 1, j)
    if orientacion == 45:
        if i < max_filas - 1 and j < max_columnas - 1:
            n1 = (i + 1, j + 1)
        if i > 0 and j > 0:
            n2 = (i - 1, j - 1)
    if orientacion == 90:
        if j > 0:
            n1 = (i, j - 1)
        if j < max_columnas - 1:
            n2 = (i, j + 1)
    if orientacion == 135:
        if j > 0 and i < max_filas - 1:
            n1 = (i + 1, j - 1)
        if j < max_columnas - 1 and i > 0:
            n2 = (i - 1, j + 1)

    return n1, n2


def get_normal_neighbours(values, orientacion, posicion_local):
    # Recuperamos posicion pixel local
    i, j = posicion_local
    max_filas, max_columnas = values.shape
    n1, n2 = 0, 0
    # Calculamos los vecinos en la normal segun la orientacion del pixel local
    if orientacion == 0:
        if j > 0:
            n1 = values[i][j - 1]
        if j < max_columnas-1:
            n2 = values[i][j + 1]
    if orientacion == 45:
        if j > 0 and i < max_filas -1:
            n1 = values[i + 1][j - 1]
        if j < max_columnas - 1 and i > 0:
            n2 = values[i - 1][j + 1]
    if orientacion == 90:
        if i < max_filas-1:
            n1 = values[i + 1][j]
        if i > 0:
            n2 = values[i - 1][j]
    if orientacion == 135:
        if i < max_filas-1 and j < max_columnas-1:
            n1 = values[i + 1][j + 1]
        if i > 0 and j > 0:
            n2 = values[i - 1][j - 1]

    return n1, n2


def noMaxSupression(grad_values, orientacion):
    filas, columnas = grad_values.shape
    edge_pixels = []
    grad_output = grad_values.copy()

    for i in range(0, filas):
        for j in range(0, columnas):
            local_value = grad_values[i][j]
            # Obtenemos los vecinos en la normal del pixel y comprobamos si el valor local es maximo
            n1, n2 = get_normal_neighbours(grad_values, orientacion[i][j], (i, j))
            if max(n1, n2, local_value) == local_value:
                # Si el punto es un maximo, lo guardamos como pixel borde
                print("maximo detectado en: "+(i, j).__str__())
                edge_pixels.append((i, j))
            else:
                # Si no se trata de un maximo, eliminamos su valor de gradiente de matriz de salida
                grad_output[i][j] = 0

    return edge_pixels, grad_output


def histerisis(gradient_values, angulos, pixeles_bordes, t_low, t_high):

    cannyEdges = []
    neighbour_list = []
    checked_pixels = np.zeros(gradient_values.shape, dtype=bool)

    for i, j in pixeles_bordes:
        # Si aun no se ha revisado ese pixel, y su gradiente es mayor al umbral alto, lo revisamos
        if not checked_pixels[i][j] and gradient_values[i][j] > t_high:
            checked_pixels[i][j] = True
            actual_position = (i, j)
            cannyEdges.append(actual_position)
            n1, n2 = get_perpendicular_neighbours(gradient_values, angulos[i][j], actual_position)
            if n1 is not None:
                neighbour_list.append(n1)
            if n2 is not None:
                neighbour_list.append(n2)
            while not gu.is_empty_list(neighbour_list):
                neighb = neighbour_list.pop(0)
                if not checked_pixels[neighb[0], neighb[1]]:
                    neighb_location = (neighb[0], neighb[1])
                    checked_pixels[neighb_location] = True
                    # Si el gradiente para el vecino es es mayor que el umbral bajo, añadimos como borde
                    if gradient_values[neighb_location] > t_low:
                        cannyEdges.append(neighb_location)
                        # Si el pixel se marca como borde, debemos revisar sus propios vecinos
                        # Los calculamos y añadimos a la lista de pendientes
                        nA, nB = get_perpendicular_neighbours(gradient_values, angulos[neighb_location],
                                                              neighb_location)
                        if nA is not None:
                            neighbour_list.append(nA)
                        if nB is not None:
                            neighbour_list.append(nB)
    return cannyEdges


if __name__ == "__main__":
    BASE_IMAGES_PATH = "/Users/javier/Documents/VA_PRACTICAS/Images"
    grayscale_image = gu.load_image(BASE_IMAGES_PATH + "/LenaRGB.jpg", gu.GRAY)
    # [gx, gy] = gradient_image(grayscale_image, 'Roberts')
    # gu.multiplot(np.add(gx, gy), gx, gy)
    edge_canny(grayscale_image, 1.4, 0.2, 0.4)
