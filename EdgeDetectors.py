import numpy as np
import Filtering as filter
import GenericUtilities as gu


def roberts_kernel():
    return np.array([[-1., 0.], [0., 1]]), np.array([[0., -1.],[1., 0.]])


def central_diff_kernel():
    return np.array([[-1, 0, 1]]), np.array([[-1],[ 0], [ 1]])


def prewitt_kernel():
    return np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]), np.array([[-1, -1, -1], [ 0,  0,  0], [ 1,  1,  1]])


def sobel_kernel():
    return np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]), np.array([[-1, -2, -1], [ 0,  0,  0], [ 1,  2,  1]])


def gradient_image(image,operator):

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
    return None


if __name__ == "__main__":

    BASE_IMAGES_PATH = "/Users/javier/Documents/VA_PRACTICAS/Images"
    grayscale_image = gu.load_image(BASE_IMAGES_PATH + "/LenaRGB.jpg", gu.GRAY)
    [gx, gy] = gradient_image(grayscale_image, 'Roberts')
    gu.multiplot(np.add(gx, gy), gx, gy)
