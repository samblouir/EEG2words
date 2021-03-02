import multiprocessing as mp
import os
from itertools import repeat

from shared_functions import load_mat, find_files

# Ensures a tmp folder is ready
os.system(f"mkdir tmp ; clear")

# raw_stem = "data_v1/task1-SR/Raw data"
raw_stem = "data_v1/task1-SR/Preprocessed"

file_paths = [file_path for file_path in find_files(raw_stem) if '_EEG.mat' in file_path ]
file_paths = sorted(file_paths)

debug = False
if debug:
    file_paths = file_paths[:2]
    print(*file_paths, sep='\n')


def load_file(index, in_file_paths, in_dict):
    file_path = in_file_paths[index]
    file_name = file_path.split('/')[-1]

    in_dict[file_name] = load_mat(file_path,
                                  standardize=True, normalize=True,
                                  normalize_range=(-1, 1), channels=[3, 4, 13],
                                  use_cache=True, flush_cache=False,
                                  debug_print=False, )


raw_dict = mp.Manager().dict()
args = zip(range(len(file_paths)), repeat(list(file_paths)), repeat(raw_dict))
mp.Pool(mp.cpu_count()).starmap(load_file, args)

for key, val in zip(raw_dict.keys(), raw_dict.values()):
    # print(f"\n {key} == \n")
    # print(f" Channels: \t 3 \t 4 \t 13")
    # print(f"{val}")
    print(f" {key}.shape == {val.shape}")
