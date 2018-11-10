import cv2
import numpy as np
import math
from matplotlib import pyplot as plt

# FLAGS PARA COLORMAP  (COLOR | ESCALA DE GRISES)
GRAY = cv2.IMREAD_GRAYSCALE
COLOR = cv2.IMREAD_COLOR
# TODO ADD BINARY IMAGE TYPE


#Print a list of values in histogram plot
def print_histogram(img, modified_img):

    plt.hist(img.ravel(), 256, [0, 1], color='b')
    plt.hist(modified_img.ravel(), 256, [0, 1], color='r')
    plt.legend(('original', 'modified'), loc='upper left')
    plt.show()
    cv2.destroyAllWindows()


#Comprueba si una lista es nula o vacia
def is_empty_list(check_list):
    if (check_list is None) or (len(check_list) == 0):
        return True
    else:
        return False


#Itera una matriz de entrada, aplicando a cada valor una funcion definida (funct)
def iterate_image(imageMatrix, funct):

    if not is_empty_list(imageMatrix):
        rows = len(imageMatrix)
        columns = len(imageMatrix[0])
    else:
        print("[ERROR] image doesn't have any information")

    #Inicializamos array para resultados de aplicar la funcion a cada valor de la imagen
    results = []
    matrix_results = np.zeros(shape=(rows, columns), dtype=np.uint8)
    for i in range(0, rows):
        for j in range(0, columns):
            # print("[DEBUG] imageMatrix"+"[" + i.__str__() + "][" + j.__str__() + "]")
            applied_fun_result = funct(imageMatrix[i][j])
            # print("[DEBUG] function result for "+"["+i.__str__()+"]["+j.__str__()+"]")
            results.append(applied_fun_result)
            matrix_results[i][j] = int(applied_fun_result)
    print_histogram(results)
    return matrix_results


#Carga una imagen de disco, ubicada en el path indicado
def load_image(path, color_type):
    image = cv2.imread(path, color_type)
    return image/255


#Muestra en pantalla una imagen cargada
def image_plot(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def multiplot(imageList):
    final_frame = cv2.hconcat((imageList[0], imageList[1], imageList[2]))
    cv2.imshow('lena', final_frame)
    cv2.waitKey(0)
    cv2.destroyAllWindows()