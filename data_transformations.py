import numpy as np


# Remaps an iterable in place, to out_min and out_max
def map_2D_array(input_iterable: list, out_min=-1, out_max=1):
    for index, sublist in enumerate(input_iterable):
        inplace_map_data(sublist, out_min=out_min, out_max=out_max)


# Normalizes an iterable in place to -1 and 1
def normalize_2D_array_inplace(input_iterable: list):
    for index, sublist in enumerate(input_iterable):
        input_iterable[index] = normalize_data(sublist)


# Standardizes an iterable in place
def standardize_2D_array_inplace(input_iterable: list):
    for index, sublist in enumerate(input_iterable):
        input_iterable[index] = standardize_data(sublist)


# Remaps each element in an iterable to a scaled range of [-1, 1]
# Derived from the famous Arduino function (https://www.arduino.cc/reference/en/language/functions/math/map/)
def inplace_map_data(input_iterable: list, out_min=-1, out_max=1) -> None:
    min_val, max_val = np.min(input_iterable), np.max(input_iterable)
    for index, element in enumerate(input_iterable):
        input_iterable[index] = (element - min_val) * (out_max - out_min) / (max_val - min_val) + out_min


# Normalizes an iterable from -1 to 1
def normalize_data(input_iterable: list, min_value=-1, max_value=1) -> np.ndarray:
    min_val, max_val = np.min(input_iterable), np.max(input_iterable)
    return_list = input_iterable - min_val
    return_list /= max_val - min_val
    return np.array(return_list)


# Standardizes an iterable
def standardize_data(input_iterable: list) -> np.ndarray:
    return_list = input_iterable - np.mean(input_iterable)
    return_list /= np.std(return_list)
    return np.array(return_list)
