import os
import multiprocessing as mp
from itertools import repeat
import numpy as np

from shared_functions import load_mat, find_files

# Ensures a tmp folder is ready
os.system(f"mkdir tmp ; clear")
print("hello!")

# raw_stem = "data_v1/task1-SR/Raw data"
raw_stem = "data_v1/task1-SR/Preprocessed"

file_paths = [file_path for file_path in find_files(raw_stem) if '_EEG.mat' in file_path and "ZGW" not in file_path]
file_paths = sorted(file_paths)
# file_paths = file_paths[:1]

print(*file_paths, sep='\n')

# import h5py
# in_path = file_paths[1]
# print(in_path)
# loaded_mat = h5py.File(in_path)
# loaded_mat=loaded_mat['EEG']
# loaded_mat=np.array(loaded_mat)
# names = [each for each in loaded_mat]
# os.system("clear")
# for name in names:
#     print(f"{name}  " * 1, f"{h5py.File(in_path)['EEG'][name]}")
#
# data = h5py.File(in_path)['EEG']['data']
# print(data)
# data=np.array(data)
# print(data)


def load_file(index, in_file_paths, in_dict):
    file_path = in_file_paths[index]
    file_name = file_path.split('/')[-1]
    # in_dict[file_name] = load_mat(file_path, standardize=True, normalize=True, normalize_range=(-1, 1), channels=[3, 4, 13], use_cached=True, debug_print=False)
    in_dict[file_name] = load_mat(file_path, standardize=True, normalize=True, normalize_range=(-1, 1), channels=[range(1)], use_cached=False, debug_print=False)


raw_dict = mp.Manager().dict()
args = zip(range(len(file_paths)), repeat(list(file_paths)), repeat(raw_dict))
args = zip(range(1), repeat(list(file_paths)), repeat(raw_dict))
mp.Pool(mp.cpu_count()).starmap(load_file, args)
# for arg in args:
#     load_file(*arg)

for key, val in zip(raw_dict.keys(), raw_dict.values()):
    print(f"\n {key} == {val}")
    # print(f"\n {key}.shape == {val.shape}")
