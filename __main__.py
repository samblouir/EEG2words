import importlib

from thread_func import run_it

# import tensorflow as tf
# import tensorflow_addons as tfa

if __name__ == "__main__":
    global data
    from shared_functions import timerD

    timerD("Started!")
    import multiprocessing as mp
    import threading as t
    import os, pickle, torch
    import ujson as json
    from itertools import repeat
    import numpy as np

    # import jax as jnp

    numless_dict_path = "data/WD/word_dict_new.dict"
    # numless_list_path = "data/WD/SR_NR2_TSR2_lower_numless_word_list.npy"

    # np.memmap(numless_list_path)
    # data = np.load(numless_list_path, allow_pickle=True, fix_imports=False)
    data = np.load(numless_dict_path, allow_pickle=True, fix_imports=False)
    timerD("loading np data")
    run_it()
    exit()

    # data = np.load(numless_dict_path, mmap_mode='r', allow_pickle=True, fix_imports=False)
    count = 0
    while True:
        print(f"\n  1. Load new_path.py")
        if count > 0:
            path = input("Data loaded. Continue?")
        count += 1

        try:
            timerD("", silent=True)
            h = t.Thread(target=run_it, args=(data,))
            h.start()
            h.join()
        except Exception as e:
            print(f"  e: {e}")

        timerD("Done!")
    timerD("Exiting...", show_total_time_elapsed=True)
