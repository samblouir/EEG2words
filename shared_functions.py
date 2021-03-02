from collections import Iterable

import h5py
import numpy as np
import os, scipy.io, json

from typing import List

from data_transformations import normalize_2D_array_inplace, standardize_2D_array_inplace, map_2D_array

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp


def load_mat(in_path: str, standardize=False, normalize=False, normalize_range=(-1, 1), channels=[-1], use_cached=True, debug_print=False) -> np.ndarray:
    restricted_chars = "/!@#$%^&*()=+[]{}\'\";:,.<>`~ "

    channel_str = '-'.join([str(cnl) for cnl in channels])
    suffix = f"{in_path}_std-{standardize}_nml-{normalize}_nml-R-{normalize_range}_cnls-{channel_str}"
    for char in restricted_chars:
        suffix = suffix.replace(char, '')
    file_type = in_path.rsplit('.', 1)[-1]
    processed_path = f"tmp/{suffix}.{file_type}.npy"

    # if the file exists, just load and return it
    if os.path.exists(processed_path) and use_cached:
        mat = np.load(f"{processed_path}")
        return mat
    else:

        # Prints out that we are working on a new file
        print(f"One-time processing {processed_path}...")
        # if we need to create the file, save it at the end
        try:
            # loaded_mat = scipy.io.loadmat(in_path)
            # Find the data's name of the Matlab dictionary,
            # assuming it's the last key...
            # tag = list(loaded_mat)[-1]
            # mat = np.array(loaded_mat[tag])
            mat = np.array(h5py.File(in_path)['EEG']['data'])
            mat=np.squeeze(mat)


        except Exception as e:
            print(f"Exception! Could not loadmat {in_path}. e: {e}")
            return None
            # mat = [-1 for _ in range(len(channels))]

        # Tries to recursively find the electrode readings by finding the deepest and largest subarray
        # while True:
        #     try:
        #         mat = max(mat, key=len)
        #         if len(mat.shape) > 1:
        #             break
        #     except:
        #         break

        # Filter to our desired channels
        if channels is None or -1 in channels:
            channels = [range(len(mat))]
        mat = mat[channels]

        # Standardize each channel independently
        if standardize:
            standardize_2D_array_inplace(mat)

        # Normalize each channel independently
        if normalize:
            map_2D_array(mat, out_min=normalize_range[0], out_max=normalize_range[1])

        # Transposes mat
        mat = [list(a) for a in zip(*mat)]

        # Print out results
        if debug_print:
            name = in_path.split("/")[-1]
            for index, reading in enumerate(mat):
                reading = list(reading)
                print(f"({name}) #{index} == ", end='\t')
                for each in reading:
                    print(f"{each:0.4f}", end='\t')
                print()

        mat = np.array(mat)
        # Saves the processed file on the disk
        f = open(processed_path, 'w')
        try:
            np.save(file=processed_path, arr=mat, )
        except Exception as e:
            print(f"Exception!: Failed to save {processed_path}. e: {e}")
            os.remove(processed_path)

        return mat


# Recursively grab all files from the input path
def find_files(in_path):
    files = []

    for file in os.listdir(in_path):
        new_path = f"{in_path}/{file}"

        if not os.path.isdir(new_path):
            files.append(new_path)

        else:
            contents = find_files(new_path)
            for each in contents:
                files.append(each)

    return files


class ActivationLayerFactory():
    def __init__(self, activation=tf.keras.activations.relu):
        self.__activation_layer_counter = 0
        self.activation = activation

    def produce(self, activation=None):
        if activation is None: activation = tf.keras.activations.relu
        temp = tf.keras.layers.Activation(activation=activation, name='activation_' + str(self.__activation_layer_counter))
        self.__activation_layer_counter += 1
        return temp


class DropoutLayerFactory():
    def __init__(self, dropout_rate=0.5):
        self.__dropout_layer_counter = 0
        self.rate = dropout_rate

    def produce(self, rate=None):
        if rate == None: rate = self.rate
        temp = tf.keras.layers.Dropout(rate=rate, name='dropout' + str(self.__dropout_layer_counter))
        self.__dropout_layer_counter += 1
        return temp


class BatchNormalizationLayerFactory():
    def __init__(self):
        self.__batchnormalization_layer_count = 0

    def produce(self):
        temp = tf.keras.layers.BatchNormalization(name='BN' + str(self.__batchnormalization_layer_count))
        self.__batchnormalization_layer_count += 1
        return temp
