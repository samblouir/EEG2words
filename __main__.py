'''
watch -n 0 'rsync -av /home/sam/PycharmProjects/EEG2words/ laura@192.168.1.15:/home/laura/PycharmProjects/EEG2words && sudo rsync -av laura@192.168.1.15:/home/laura/PycharmProjects/EEG2words/data/ /home/sam/PycharmProjects/EEG2words/data'
'''

from itertools import repeat
import ujson as json
from typing import List

from shared_functions import init, load_et_data, load_eeg_data, print_dict, timerD, prepare_embedding_layer
import numpy as np
# import jax as np
import os
import multiprocessing as mp

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import tensorflow as tf


def prepare_glove(name="6B"):
    # Store the current files
    curr_files = os.listdir()

    os.system("mkdir -p data/glove")
    if len(os.listdir(f"data/glove")) < 3:
        os.system("wget http://nlp.stanford.edu/data/glove.6B.zip")
    os.system(f"mv glove.{name}.zip data/glove")
    os.system(f"unzip data/glove/glove.{name}.zip")

    new_files = [file for file in os.listdir() if file not in curr_files]
    for file in new_files:
        os.system(f"mv {file} data/glove")


def q_smp_N(func, args, N):
    return mp.Pool(N).starmap(func, args)


def q_smp(func, args):
    return mp.Pool(mp.cpu_count()).starmap(func, args)
    # return mp.Pool(mp.cpu_count()).starmap(func, args)
    # return mp.Pool(mp.cpu_count()).starmap(func, prepare_arguments(*args))


def mp_load_file(glove_file, index, in_dict):
    line: str = glove_file[index]
    data = line.split()
    word, vector = data[0], np.array(data[1:], dtype=np.float32)
    in_dict[word] = vector


# def mp_load_file_two_arrays(glove_file, index, n, in_words_list, in_vectors_list):
def mp_load_file_two_arrays(glove_file, index, N, in_words_list):
    line: str = glove_file[index]
    data = line.split()
    word, vector = data[0], np.array(data[1:N + 1], dtype=np.float32)
    # in_words_list.append([word, vector])
    in_words_list.append(word)


# def mp_load_file_two_arrays2(glove_file, proc_count, n, in_words_list, in_vectors_list):
def mp_load_file_two_arrays2(glove_file, proc_count, N):
    batch_size = int(len(glove_file) / N)
    lines = glove_file[proc_count * batch_size:(proc_count + 1) * batch_size]
    words = ["" for _ in range(len(lines))]
    vectors = [[0.0 for __ in range(len(lines[0].split()) - 1)] for _ in range(len(lines))]
    words, vectors = np.array(words, dtype=object), np.zeros_like(vectors, dtype=np.float32)

    for index, line in enumerate(lines):
        data = line.split()
        word, vector = data[0], np.array(data[1:], dtype=np.float32)
        # print(f"word/vector == {word} / {vector}")
        words[index] = word
        # words[index] = word[:]
        vectors[index] = vector[:]
        # vectors[index] = vector

    # print(words, vectors)
    return np.squeeze(words), np.squeeze(vectors)
    # return words,vectors


def slice_file(glove_file, index, N):
    line: str = glove_file[index]
    data = line.split()
    return ' '.join(data[:N + 1])


def prepare_arguments(*x):
    args = []
    for index in range(len(x)):
        try:  # A more precise check, but expensive check
            iter(x[index])
            args.append(x[index])
        except:
            args.append(repeat(x[index]))
    return zip(*args)


def load_embeddings(n=50):
    new_folder = f"data/glove/6B.{n}d"
    os.system(f"mkdir -p {new_folder}")
    vector_path = f"{new_folder}/glove.6B.{n}d.vectors.npy"
    word_path = f"data/glove/glove.6B.300d.words.npy"
    glove_path = f"data/glove/glove.6B.300d.txt"

    ###################################################################################################
    ###################################################################################################
    if not os.path.exists(vector_path) or not os.path.exists(word_path):
        glove = open(glove_path).readlines()

        process_count = mp.cpu_count()
        args = zip(repeat(glove), range(process_count), repeat(process_count))
        results = q_smp_N(mp_load_file_two_arrays2, args, process_count)

        # Combine MP results and split into word and vector lists
        a, b = zip(*results)
        a, b = np.concatenate([*a]), np.concatenate([*b])
        a, b = zip(*sorted(((zip(a, b)))))
        word_array, vector_array = np.array(a), np.array(b)

        if not os.path.exists(vector_path): np.save(vector_path, vector_array, allow_pickle=True)
        if not os.path.exists(word_path): np.save(word_path, word_array, allow_pickle=True)

    ###################################################################################################
    ###################################################################################################

    a = np.load(word_path, allow_pickle=True)
    b = np.load(vector_path, allow_pickle=True)
    # Create a dict with pointers to b
    c = {str(a[idx]): b[idx] for idx in range(len(a))}

    word_array, vector_array, word_vector_dict = np.array(a), np.array(b), c
    timerD("Loading the data")
    return word_array, vector_array, word_vector_dict


# @tf.function(experimental_compile=True)
# @tf.function()
def do_it_all(values=None):
    pass
    # allbig = tf.repeat(all, repeats=len(values))
    # a = tf.reduce_sum([v for v in values])
    # b = tf.abs
    # c = tf.subtract
    # d = a(b(c())
    # for v in values:
    # distance = a(b(c(v, all)), axis=2)
    # distance = tf.reduce_sum(tf.abs(tf.subtract(v, all)), axis=2)


if __name__ == '__main__':
    timerD("Importing libraries")

    words = ["car", "automobile", "truck", "bus", "limo", "jeep", "boat", "canoe", "dinghy", "motorboat", "yacht", "catamaran"]

    n = 300
    a, b, c = load_embeddings(n=n)
    word_list, vector_list, word_vector_dict = a, b, c

    timerD(f"load_embeddings({n}) complete")
    # all = tf.expand_dims(list(embeddings_dict.values()), 1)

    # timerD("Entering do_it_all")
    # do_it_all(embeddings_dict.items())
    # timerD("Completed do_it_all")

    # print(f"embeddings_dict = {embeddings_dict.keys()[keys]}")
    # print()
    # all = tf.expand_dims(list(embeddings_dict.values()), 1)
    # for k, v in word_dict.items():
    # for k, v in embeddings_dict.items():
    # print(v, k)
    # print(f"{word}: {np.mean(embeddings_dict[word])}")
    # distance = tf.reduce_sum(tf.abs(tf.subtract(v, all)), axis=2)
    # distance = tf.reduce_sum(tf.abs(tf.subtract(v, tf.expand_dims(list(word_dict.values()), 1))), axis=2)
    # timerD(f"distances from {k}: {distance}")
    # timerD(f"distances from {k}")

    timerD("Total time taken", show_total_time_elapsed=True)
    # Ensure expected folders are created
    # init()
    #
    # # Loads et data
    # et_dict = load_et_data()
    #
    # channel_indices = [3, 4, 13]
    # eeg_dict = load_eeg_data(channel_indices)

    # print_dict(eeg_dict)
    # print_dict(et_dict)
