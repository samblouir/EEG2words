import h5py
import scipy.io as io
import os, scipy.io, json

import numpy as np
import multiprocessing as mp
import os
from itertools import repeat
import pandas as pd

from shared_functions import load_mat, find_files, convert_file

# Ensures a tmp folder is ready for data_v1 and data_v2
os.system(f"mkdir tmp2")
os.system(f"clear")

# raw_stem = "data_v1/task1-SR/Raw data"
raw_stem = "data_v2/task1-NR/Preprocessed"

# file_paths = [file_path for file_path in find_files(raw_stem) if '_EEG.mat' in file_path]
file_paths = [file_path for file_path in find_files(raw_stem) if '_EEG.mat' in file_path]
file_paths = sorted(file_paths)

debug = True
debug = False
if debug:
    file_paths = file_paths[:2]
    print(*file_paths, sep='\n')


# def load_file(index, in_file_paths, in_dict):
#     os.system("clear")
#     file_path = in_file_paths[index]
#     file_name = file_path.split('/')[-1]
#     # in_dict[file_name] = np.load(file_path)
#
#     contents = convert_file(file_path,
#                             standardize=True, normalize=True,
#                             # normalize_range=(-1, 1), channels=[-1], # -1 just loads all channels
#                             normalize_range=(-1, 1), channels=[3, 4, 13],
#                             use_cache=True, flush_cache=False,
#                             debug_print=True, )
#     # debug_print=False, )
#
#     if contents is not None:
#         in_dict[file_name] = contents
#         # in_dict[file_name].append(contents)
#         # in_dict.append(contents)


def csv_file(index, in_file_paths, in_dict):
    file_path = in_file_paths[index]
    file_name = file_path.split('/')[-1]
    output_file = f"processed/{file_name}.csv"


    file = h5py.File(file_path, 'r')
    file = file['EEG']['data']
    new_arr = file
    new_arr = np.squeeze(new_arr)
    new_arr = pd.DataFrame(new_arr)
    new_arr = np.squeeze(new_arr)
    in_dict[file_name] = new_arr

    print(f"Opened {file_name}! ({index}/{len(in_file_paths)})")


if __name__ == '__main__':
    raw_dict = mp.Manager().dict()
    args = zip(range(len(file_paths)), repeat(list(file_paths)), repeat(raw_dict))
    mp.Pool(mp.cpu_count()).starmap(csv_file, args)

    for key, value in zip(raw_dict.keys(), raw_dict.values()):
        print(f"{key} == {value}")


    exit()
###########

# import scipy.io as io
#
# import numpy as np
# import multiprocessing as mp
# import os
# from itertools import repeat
#
# from shared_functions import load_mat, find_files, convert_file
#
# import convert_v2
# exit()
#
# # Ensures a tmp folder is ready for data_v1 and data_v2
# os.system(f"mkdir tmp tmp2")
# os.system(f"clear")
#
# # raw_stem = "data_v1/task1-SR/Raw data"
# raw_stem = "data_v1/task1-SR/Preprocessed"
#
# file_paths = [file_path for file_path in find_files(raw_stem) if '_EEG.mat' in file_path]
# file_paths = sorted(file_paths)
#
# debug = False
# if debug:
#     file_paths = file_paths[:2]
#     print(*file_paths, sep='\n')
#
#
# def load_file(index, in_file_paths, in_dict):
#     file_path = in_file_paths[index]
#     file_name = file_path.split('/')[-1]
#     # in_dict[file_name] = np.load(file_path)
#
#     contents = load_mat(file_path,
#                         standardize=True, normalize=True,
#                         # normalize_range=(-1, 1), channels=[-1], # -1 just loads all channels
#                         normalize_range=(-1, 1), channels=[3, 4, 13],
#                         use_cache=True, flush_cache=False,
#                         debug_print=True, )
#                         # debug_print=False, )
#
#     if contents is not None:
#         in_dict[file_name] = contents
#
#
# raw_dict = mp.Manager().dict()
# args = zip(range(len(file_paths)), repeat(list(file_paths)), repeat(raw_dict))
#
# for arg in args:
#     convert_file(*arg)
#     break
#
# # mp.Pool(mp.cpu_count()).starmap(load_file, args)
#
# for key, val in zip(raw_dict.keys(), raw_dict.values()):
#     # print(f"\n {key} == \n")
#     # print(f" Channels: \t 3 \t 4 \t 13")
#     # print(f"{val}")
#     print(f" {key}.shape == {val.shape}")
#
# first = raw_dict[raw_dict.keys()[0]]
#
# # print(first.shape)
#
# # first = list(first)
#
# # first.tofile('hi.csv', sep=',', format='%2.2f')
# # np.savetxt("lol.csv", first, delimiter=',', )
# import pandas as pd
#
# # pd.DataFrame(data=first).to_csv("pd.csv")
# # pd.DataFrame(data=first, columns=['a','b','c']).to_csv("pd.csv")
# # f=open('lol.csv','w')
# # for each in first:
# #     np.csv.write(f)
# #     f.write(each)
#
# # os.system("chmod 757 *")
