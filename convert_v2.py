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


def csv_file(index, in_file_paths):
    file_path = in_file_paths[index]
    file_name = file_path.split('/')[-1]
    print(f"Working on {file_name}...")

    # in_lock.acquire()
    file = h5py.File(file_path, 'r')
    file = file['EEG']['data']
    new_arr = file
    new_arr = np.squeeze(new_arr)
    lol = pd.DataFrame(new_arr)
    # in_lock.release()
    output_file = f"processed/{file_name}.csv"
    print(f"Saving {output_file}...")
    # in_lock.acquire()
    lol.to_csv(output_file, header=False, index=False, compression='bz2')
    # in_lock.release()
    # os.system(f"chmod 757 {output_file}")

if __name__ == '__main__':

    # raw_dict = mp.Manager().dict()
    # args = zip(range(len(file_paths)), repeat(list(file_paths)), repeat(raw_dict))
    args = zip(range(len(file_paths)), repeat(list(file_paths)))
    # args = zip(range(2), repeat(list(file_paths)), repeat(mp_lock))
    # args = zip(range(1), repeat(list(file_paths)))
    # for arg in args:
    #     csv_file(*arg)
    #     exit()
    mp.Pool(mp.cpu_count()).starmap(csv_file, args)

    exit()

p0 = "data_v2/task1-NR/Preprocessed/YAC/bip_YAC_NR5_EEG.mat"
# p0 = "data_v2/task1-NR/RawData/YAC/YAC_NR1_EEG.mat"
# p0 = "data_v2/task1-NR/RawData/YAC/YAC_NR1_ET.mat"
p1 = "data_v2/task1-NR/Preprocessed/YAC/bip_YAC_NR6_EEG.mat"
# import tables
os.system(f"mkdir processed")
n0 = p0.rsplit("/", 1)[1]
dest_path = f"processed/{n0}.xml"
cmd = f"h5dump -x {p0} > {dest_path}"

if not os.path.exists(dest_path):
    print(f"Creating {dest_path}...")
    os.system(cmd)

hiya = []
hd = {}


def add2hiya(content, z):
    global hd
    hd[content] = np.squeeze(z)
    # hiya.append(z)
    # tl = []
    # for each in list(z):
    #     each = np.squeeze(each)
    #     tl.append(each)

    # print(f"{content} \t\t {tl}")


# os.system("chmod -R 757 data_v2/*")
# os.system("ls -al data_v2/task1-NR/RawData/YAC")


os.system("clear")
is_mat73 = True
try:
    h5py.File(p0, 'r')
except:
    is_mat73 = False
    print(f"is_mat73 == {is_mat73}")

if is_mat73:
    file = h5py.File(p0, 'r')
    file = file['EEG']['data']
    new_arr = file
    new_arr = np.squeeze(new_arr)
    lol = pd.DataFrame(new_arr)
    print(lol)
    lol.to_csv("processed/hi.csv", header=False, index=False)

    exit()
    for each in hiya:
        print(each, '\n')
    for k, v in zip(hd.keys(), hd.values()):
        print(f"{k} \t {v}")
else:
    print("yeah!")
    exit()

    file = scipy.io.loadmat(p0)
    tags = list(file)
    # assuming it's the last key...
    tag = tags[-1]
    data = np.array(file[tag])

    print(data)

# from xml.etree import ElementTree as ET
#
# tree = ET.parse(dest_path)
# print(tree.getroot())
# hi = tree.iter()
# for each in hi:
#     print(each)
# print(tree.iter())
exit()

# for each in raw_dict:
#     print(f"{each} \n ")
# lol = h5py.File()
for key, value in zip(raw_dict.keys(), raw_dict.values()):

    # for each in key:
    #     print(f"{key} \t {each}")
    # print(f"{key} : {value}")
    for each in value:
        print(f"{key} \t  \t {each[0]}")
        # print(f" \t {each[0]}")
exit()

# mp.Pool(mp.cpu_count()).starmap(load_file, args)

for key, val in zip(raw_dict.keys(), raw_dict.values()):
    # print(f"\n {key} == \n")
    # print(f" Channels: \t 3 \t 4 \t 13")
    # print(f"{val}")
    print(f" {key}.shape == {val.shape}")

first = raw_dict[raw_dict.keys()[0]]

# print(first.shape)

# first = list(first)

# first.tofile('hi.csv', sep=',', format='%2.2f')
# np.savetxt("lol.csv", first, delimiter=',', )
import pandas as pd

# pd.DataFrame(data=first).to_csv("pd.csv")
# pd.DataFrame(data=first, columns=['a','b','c']).to_csv("pd.csv")
# f=open('lol.csv','w')
# for each in first:
#     np.csv.write(f)
#     f.write(each)

# os.system("chmod 757 *")
