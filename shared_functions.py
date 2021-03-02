from collections import Iterable

import h5py
import numpy as np
import os, scipy.io, json

from typing import List

from data_transformations import normalize_2D_array_inplace, standardize_2D_array_inplace, map_2D_array


def load_mat(in_path: str, standardize=False, normalize=False, normalize_range=(-1, 1), channels=[-1],
             use_cache=True, flush_cache=False,
             debug_print=False) -> np.ndarray:
    restricted_chars = "/!@#$%^&*()=+[]{}\'\";:,.<>`~ "

    # Makes a unique name, per our arguments/settings
    channel_str = '-'.join([str(cnl) for cnl in channels])
    suffix = f"{in_path}_std-{standardize}_nml-{normalize}_nml-R-{normalize_range}_cnls-{channel_str}"
    for char in restricted_chars:
        suffix = suffix.replace(char, '')
    file_type = in_path.rsplit('.', 1)[-1]
    processed_path = f"tmp/{suffix}.{file_type}.npy"

    # If the file exists, just load and return it
    if os.path.exists(processed_path) and use_cache and not flush_cache:
        mat = np.load(f"{processed_path}")
        return mat
    else:

        # Prints out that we are working on a new file
        print(f"One-time processing {processed_path}...")
        # if we need to create the file, we can load it, process it, and save it at the end
        try:

            # Runs on raw data files
            # loaded_mat = scipy.io.loadmat(in_path)
            # Find the data's name of the Matlab dictionary,
            # assuming it's the last key...
            # tag = list(loaded_mat)[-1]
            # mat = np.array(loaded_mat[tag])

            # Runs on non-raw files
            mat = np.array(h5py.File(in_path)['EEG']['data'])
            mat = np.squeeze(mat)


        except Exception as e:
            print(f"Exception! Could not load {in_path}. e: {e}")
            return None
            # mat = np.array([-1 for _ in range(len(channels))])
            # return mat

        # Runs on raw data files
        # Tries to recursively find the electrode readings by finding the deepest and largest subarray
        # while True:
        #     try:
        #         mat = max(mat, key=len)
        #         if len(mat.shape) > 1:
        #             break
        #     except:
        #         break

        # Transposes mat
        mat = np.array([np.array(a) for a in zip(*mat)])

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

        if use_cache:
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
