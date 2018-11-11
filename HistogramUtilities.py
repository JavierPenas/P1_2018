import GenericUtilities as gu
import numpy as np


#ENVIROMENTAL VARIABLES
BASE_IMAGES_PATH = "/Users/javier/Documents/VA_PRACTICAS/Images"

def adjust_intensity(image, in_range, out_range):

    if not gu.is_empty_list(image):
        (rows, columns) = image.shape[:2]
        i_min, i_max = __set_in_range__(in_range, image)
        o_min, o_max = __set_out_range__(out_range)
        ranges = {'IMIN': i_min, 'IMAX': i_max, 'OMIN': o_min, 'OMAX': o_max}

        output = np.zeros((rows, columns), dtype="float32")
        for y in np.arange(0, rows):
            for x in np.arange(0, columns):
                output[y, x] = adjust_intensity_fun(image[y][x], ranges)
        return output
    else:
        print("[ERROR] image doesn't have any information")
        return None


#Funcion modificacion rango dinamico
def adjust_intensity_fun(value,ranges):
    output = ranges.get("OMIN") + ((ranges.get("OMAX")-ranges.get("OMIN"))*(value-ranges.get("IMIN")))/(ranges.get("IMAX")-ranges.get("IMIN"))
    return output


def __set_in_range__(in_range, image):

    if not gu.is_empty_list(in_range):
        if len(in_range) == 2:
            return in_range[0], in_range[1]
        else:
            print("[ERROR] Debe especificar dos o ningún valor para el rango de entrada")
    else:
        return np.min(image), np.max(image)


def __set_out_range__(out_range):

    if not gu.is_empty_list(out_range):
        if len(out_range) == 2:
            return out_range[0], out_range[1]
        else:
            print("[ERROR] Debe especificar dos o ningún valor para el rango de salida")
    else:
        return 0, 1


def create_histogram(image, nBins):
    image = image * (nBins-1)
    hist = np.zeros((1, nBins))
    filas, columnas = image.shape
    for i in np.arange(0, filas):
        for j in np.arange(0, columnas):
            hist[0, image[i, j].astype("int")] += 1
    return hist


def equalizeFun(g, histogram, shape):
    g = g.astype("int")
    filas, columnas = shape
    acumulative = 0

    for i in range(0, g):
        acumulative += histogram[0, i]

    return (acumulative*255)/(filas*columnas)


def equalizeIntensity(image, nBins):

    if nBins is None:
        nBins = 256
    filas, columnas = image.shape
    output = np.zeros((filas, columnas), dtype="float32")
    histogram = create_histogram(image, nBins)

    for i in np.arange(0, filas):
        for j in np.arange(0, columnas):
            output[i, j] = equalizeFun(image[i, j]*(nBins-1), histogram, (filas, columnas))

    return output


if __name__ == "__main__":
    grayscale_image = gu.load_image(BASE_IMAGES_PATH + "/LenaRGB.jpg", gu.GRAY)
    output_img = equalizeIntensity(grayscale_image, None)
    gu.image_plot(output_img)
