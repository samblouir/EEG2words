import multiprocessing as mp
import os
import time
from itertools import repeat
from typing import List
import fasttext
import fasttext.util
import h5py, scipy.io
import numpy as np
import json

# Where to recursively search for our files
# raw_stem = "data_v1/task1-SR/Raw data"
# raw_stem = "data_v2/task1-NR/Preprocessed"
raw_stem = "data_v2/task1-NR/Preprocessed/YAC"


# Initialize expected directories
def init():
    if not os.path.exists("processed"):
        os.system("mkdir processed")
    if not os.path.exists("et_processed"):
        os.system("mkdir et_processed")
    os.system("clear")


# Prints a dictionary
def print_dict(in_dict: dict):
    for key, value in zip(in_dict.keys(), in_dict.values()):
        print("*" * 40)
        print(f"{key} == \n {value} \n ")


# Gets the file path to the cached file
def get_cached_file_path(file_name: str, channel_indices=None, prefix="processed") -> str:
    if channel_indices is None:  # Avoids mutable object as default argument
        channel_indices = [0]
    return f"{prefix}/{file_name}-{'_'.join(map(lambda x: str(x), channel_indices))}.csv.npy"


# If the file does not exist or force_cache == True,
# load in the file and save it to the cache
def cache_file(file_to_cache: str, cache_path: str,
               channel_indices: List[int], force_cache=False) -> None:
    if not os.path.exists(cache_path) or force_cache:
        file = h5py.File(file_to_cache, 'r')
        file = file['EEG']['data']
        new_arr = np.squeeze(file)
        new_arr = np.transpose(new_arr)
        new_arr = new_arr[channel_indices]
        new_arr = np.array([np.array(a) for a in zip(*new_arr)])
        output_file = cache_path.rsplit(".npy", 1)[0]
        np.save(file=output_file, arr=new_arr)
        if os.name == "posix":
            os.system(f"chmod 757 {cache_path}")


# Opens and loads a file and ensures a cached version is created
def cache_open(index, in_file_paths, in_dict, channel_indices) -> None:
    # Calculate our file paths
    file_path = in_file_paths[index]
    file_name = file_path.split('/')[-1]
    cached_file_path = get_cached_file_path(file_name, channel_indices)

    # Creates a relevant cache file, if it does not exist.
    cache_file(file_to_cache=file_path, cache_path=cached_file_path,
               channel_indices=channel_indices, force_cache=False, )

    # Load in from the relevant cache file
    new_arr = np.load(cached_file_path, mmap_mode='r', allow_pickle=True)

    # Filter to the relevant indices
    # if channel_indices is not None:
    #     new_arr = new_arr[channel_indices]

    # Store it in the multiprocessing dictionary
    in_dict[file_name] = new_arr

    # A little victory message
    # print(f'Opened {file_name}!')


# Caches and loads EEG data into an mp.dict() using multiprocessing (mat version 7.3)
def load_eeg_data(channel_indices=None) -> dict:
    # Find all _EEG.mat files
    file_paths = [file_path for file_path in find_files(raw_stem) if '_EEG.mat' in file_path]
    # Create an mp dict
    out_dict = mp.Manager().dict()

    # Use mp.Pool on cache_open()
    args = zip(
        range(len(file_paths)), repeat(list(file_paths)),
        repeat(out_dict), repeat(channel_indices), )
    mp.Pool(mp.cpu_count()).starmap(cache_open, args)

    # Return the EEG data as a dict
    return out_dict


# Caches and loads EEG data into an mp.dict() using multiprocessing (mat version <7.3)
def cache_open_et(index, file_paths, in_dict):
    file_path = file_paths[index]
    file_name = file_path.split('/')[-1]
    file = scipy.io.loadmat(file_path)
    cached_file_path = get_cached_file_path(file_name, prefix="et_processed")

    renew_cache = False
    renew_cache = True
    # try:
    #     with open(cached_file_path, 'r') as fin:
    #         temp_dict = json.load(fin)
    #     in_dict[file_name] = temp_dict
    # except:
    #     renew_cache = True

    if not os.path.exists(cached_file_path) or renew_cache:
        #  The list of six relevant keys
        #  ['raw_comments', 'colheader', 'data', 'messages', 'eyeevent', 'event']
        keys = list(file.keys())[-6:]

        #  Loads our six relevant values
        raw_comments = file[keys[0]]
        colheader = file[keys[1]]
        data = file[keys[2]]
        messages = file[keys[3]]
        eyeevent = file[keys[4]]
        event = file[keys[5]]

        #  Prepare all key and values
        raw_comments = process_text_array(raw_comments)
        colheader = process_text_array(colheader)
        data = data
        messages = process_text_array(messages)
        eyeevent = np.squeeze(eyeevent)

        #  Prepare all key and values
        temp_dict = {}
        temp_dict['raw_comments'] = raw_comments
        temp_dict['colheader'] = colheader
        temp_dict['data'] = data
        temp_dict['messages'] = messages
        temp_dict['eyeevent'] = eyeevent
        temp_dict['event'] = event

        #   SAMPLES	GAZE	LEFT	HTARGET	RATE	 500.00	TRACKING	CR	FILTER	2
        msg = messages.splitlines()
        msgs = []
        ESACC_reached = False
        for line in msg:
            # t0 = line.replace("\t",'')
            t0 = line.split()

            if not ESACC_reached and "ESACC" in line:
                ESACC_reached = True

            if ESACC_reached and "ESACC" in line:
                msgs.append(t0)

        labels = data[:, -1]
        data = data[:, :-1]
        # msgs = [m.split('\t') for m in msg]
        print(*msgs, sep='\n')

        # for key in keys:
        #     curr_data = temp_dict[key]

        #  https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable
        # def default(obj):
        #     if type(obj).__module__ == np.__name__:
        #         if isinstance(obj, np.ndarray):
        #             return obj.tolist()
        #         else:
        #             return obj.item()
        #     raise TypeError('Unknown type:', type(obj))

        #  json-serializes and writes to cached_file_path
        # with open(cached_file_path, 'w') as fout:
        #     print(json.dumps(data, default=default), file=fout)

        #  Linux check for proper permissions
        if os.name == "posix":
            os.system(f"chmod 757 {cached_file_path}")

        #  Makes our dict's file_name value point to our dict
        in_dict[file_name] = temp_dict

    #  A little victory message
    # print(f'Opened {file_name}!')


# Caches and loads ET data into an mp.dict() using multiprocessing (mat v7.3)
def load_et_data() -> dict:
    # Find all _ET.mat files
    file_paths = [file_path for file_path in find_files(raw_stem) if '_ET.mat' in file_path]
    # Create an mp dict
    out_dict = mp.Manager().dict()

    # Use mp.Pool on cache_open()
    args = zip(
        range(len(file_paths)), repeat(list(file_paths)),
        repeat(out_dict), )
    mp.Pool(mp.cpu_count()).starmap(cache_open_et, args)

    # Return the EEG data as a dict
    return out_dict


# Combines and returns as a string a joining of arrays of arrays of strings
def process_text_array(input_raw_comments, sep='\n'):
    # Assemble all comment lines into a list
    comments_list = []
    for raw_line in input_raw_comments:
        for line in raw_line:
            line = str(np.squeeze(line)).strip()
            comments_list.append(line)

    # Join the list, each line separated by sep
    return sep.join(comments_list)


# Recursively grab all files from the input_raw_comments path
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


# Keeps track of time, easily.
# Pass in a String for a pretty print. Returns total time elapsed in seconds.
def timerD(name="Last job", silent=False, show_total_time_elapsed=False, start_timer=time.perf_counter(), return_difference=True, reset=False, end='\n'):
    if "last_time" not in timerD.__dict__: timerD.last_time = start_timer
    if reset or "counter" not in timerD.__dict__: timerD.counter = 0

    new_time = time.perf_counter()
    difference = new_time - timerD.last_time
    timerD.last_time = new_time

    if not silent:
        if show_total_time_elapsed:
            print(f"  {timerD.counter:3d}) {time.perf_counter() - start_timer:0.3f}s (total) \t {name} ", end=end)
        else:
            print(f"  {timerD.counter:3d}) {difference:0.3f}s \t\t {name} ", end=end)
        timerD.counter += 1

    if return_difference:
        return difference
    return time.perf_counter() - start_timer


# Adapted from https://github.com/aneesh-joshi/LSTM_POS_Tagger/blob/master/make_model.py
def prepare_embedding_layer(input_dim, embed_out_dim, word_list, load_embeddings=True, language='en'):
    new_depth = embed_out_dim

    fasttext.util.download_model(language)
    curr_file = f'cc.{language}.300.bin'
    desired_file = f"tmp/{curr_file[:5]}.{new_depth}.bin"
    if not os.path.exists(desired_file):
        ft = fasttext.load_model(f"data/{curr_file}")
        fasttext.util.reduce_model(ft, embed_out_dim)
        ft.save_model(desired_file)
        timerD(f"Caching fasttext at {curr_file[4:6]}-{embed_out_dim} dimensions")
    ft = fasttext.load_model(desired_file)
    timerD(f"Loading fasttext-{embed_out_dim}")

    embedding_weights = np.random.random((input_dim, embed_out_dim))

    if load_embeddings:
        for index, word in enumerate(word_list.keys()):
            embedding_weights[index] = ft.get_word_vector(word)
        timerD(f"Loading embedding_weights")

    return embedding_weights
