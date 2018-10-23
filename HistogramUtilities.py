import GenericUtilities as gu
import Variables as var
import numpy as np


def adjust_intensity(image, inRange, outRange):

    #Calculamos los límites del nuevo histograma
    #Los valores de estas variables se almacenan en Variables.py
    __set_in_range__(inRange, image)
    __set_out_range__(outRange)
    #Aplicamos la funcion de ajuste a cada pixel de la imagen. (Iteracion + Aplicacion)
    res_image = gu.iterate_image(image, mod_range_fun)
    # Podemos mostrar la imagen resultante si lo deseamos
    # gu.image_plot(res_image)


#Funcion modificacion rango dinamico
def mod_range_fun(value):
    # print("[DEBUG] This is histogram function")
    outputValue = var.GMIN_NORM + ((var.GMAX_NORM-var.GMIN_NORM)*(value-var.GMIN))/(var.GMAX-var.GMIN)
    return int(outputValue)


# def transform_function(image):
#     rows = len(imageMatrix)
#     columns = len(imageMatrix[0])
#     transformed_hist = (accumulative_hist(image) / (rows*columns)) * 255
#     return transformed_hist


#LOCAL FUNCTIONS FOR GLOBALS DEFINITION
def __set_in_range__(inRange,image):

    if gu.is_empty_list(inRange):
        var.GMAX = np.max(image)
        var.GMIN = np.min(image)
    else:
        if (len(inRange) < 2) or (np.min(inRange) < 0):
            print("[ERROR] Unexpected error: inRange values must be size 2 instead of 1")
        else:
            var.GMAX = inRange[1]
            var.GMIN = inRange[0]


def __set_out_range__(outRange):

    if gu.is_empty_list(outRange):
        var.GMAX_NORM = 1
        var.GMIN_NORM = 0
    else:
        if (len(outRange) < 2) or (np.min(outRange) < 0):
            print("[ERROR] Unexpected error: inRange values must be size 2 instead of 1")
        else:
            var.GMAX_NORM = outRange[1]
            var.GMIN_NORM = outRange[0]
