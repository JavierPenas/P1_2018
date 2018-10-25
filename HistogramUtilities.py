import GenericUtilities as gu
import numpy as np


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
    return int(output)


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
