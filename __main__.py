'''
watch -n 0 'rsync -av /home/sam/PycharmProjects/EEG2words/ laura@192.168.1.15:/home/laura/PycharmProjects/EEG2words && sudo rsync -av laura@192.168.1.15:/home/laura/PycharmProjects/EEG2words/data/ /home/sam/PycharmProjects/EEG2words/data'
'''

from itertools import repeat
import ujson as json
from typing import List

from shared_functions import init, load_et_data, load_eeg_data, print_dict, timerD, prepare_embedding_layer
import numpy as np
# import jax as jnp
import os
import multiprocessing as mp

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import tensorflow as tf
from tensorflow.python.layers.base import Layer

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)


def prepare_glove(name="6B"):
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


def q_mp(func, arg):
    return mp.Pool(mp.cpu_count()).map(func, arg)


def mp_load_file(glove_file, index, in_dict):
    line: str = glove_file[index]
    data = line.split()
    word, vector = data[0], np.array(data[1:], dtype=np.float32)
    in_dict[word] = vector


def mp_load_file_two_arrays(glove_file, index, N, in_words_list):
    line: str = glove_file[index]
    data = line.split()
    word, vector = data[0], np.array(data[1:N + 1], dtype=np.float32)
    # in_words_list.append([word, vector])
    in_words_list.append(word)


def mp_load_file_two_arrays2(glove_file, proc_count, N):
    batch_size = int(len(glove_file) / N)
    lines = glove_file[proc_count * batch_size:(proc_count + 1) * batch_size]
    words = ["" for _ in range(len(lines))]
    vectors = [[0.0 for __ in range(len(lines[0].split()) - 1)] for _ in range(len(lines))]
    words, vectors = np.array(words, dtype=object), np.zeros_like(vectors, dtype=np.float32)

    for index, line in enumerate(lines):
        data = line.split()
        word, vector = data[0], np.array(data[1:], dtype=np.float32)
        words[index] = word
        vectors[index] = vector
        # vectors[index] = vector[:]

    return np.squeeze(words), np.squeeze(vectors)


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


def load_embeddings(n=50, glove_size=6):
    new_folder = f"data/glove/{glove_size}B.{n}d"
    os.system(f"mkdir -p {new_folder}")
    vector_path = f"{new_folder}/glove.{glove_size}B.{n}d.vectors.npy"
    word_path = f"data/glove/glove.{glove_size}B.300d.words.npy"
    glove_path = f"data/glove/glove.{glove_size}B.300d.txt"

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
    return word_array, vector_array, word_vector_dict


def get_file(URL, path="data"):
    file_name = URL.rsplit("/", 1)[1]
    if not os.path.exists(file_name):
        os.system(f"wget {URL}")
        os.system(f"mv {file_name} {path}/")
    return f"{path}/{file_name}"


def get_work_load(in_file, N=mp.cpu_count()):
    batch_size = int(len(in_file) / N)
    return in_file[N * batch_size:(N + 1) * batch_size]


def mp_split(array, index):
    return array[index].split()


def mp_split_vectors(array, index, in_dict):
    each = array[index].split()
    return [in_dict[e] for e in each]


def get_vectors(array, in_dict):
    return [in_dict[e.lower()] for e in array]


def get_index_vectors(array, index, in_dict):
    # array = get_work_load(array)
    # words = [w.lower() for w in array]
    # return [in_dict[e] for e in words]

    return [in_dict[w.lower()] for w in array[index].split()]


def get_analogy_vector(array, index, in_dict):
    # words = [a.split() for a in array[index]]
    vectors = [in_dict[w.lower()] for w in array[index].split()]
    result = vectors[1] - vectors[0] + vectors[2]

    # words = array[index].split()
    # vectors = [in_dict[w.lower()] for w in words]
    # result = vectors[1] - vectors[0] + vectors[2]
    return result


class AnalogyLayer(Layer):
    def __init__(self):
        super(AnalogyLayer, self).__init__()
        self.counter = 0

    @tf.function(experimental_compile=True)
    def call(self, inputs, training=False):
        # Like a TF functional model
        x = tf.subtract(inputs[0], inputs[1])
        x = tf.abs(x)
        x = tf.reduce_sum(x, axis=1)
        x = tf.argmin(x)  # Can be easily changed to grab the K-nearest
        # print(self.counter, end='\r')
        # self.counter += 1
        return x


# @tf.function()
# @tf.function(experimental_compile=True)
counter = 0
def do_it_all(curr=None, b=None):
    global counter
    inputs = tf.subtract(curr, b)
    x = tf.subtract(curr, b)
    x = tf.abs(x)
    x = tf.reduce_sum(x, axis=1)
    x = tf.argmin(x)
    print(counter, end='\r')
    counter += 1
    return x


def convert_to_tensors(curr):
    return tf.convert_to_tensor(curr, dtype=tf.float64)


def load_analogies(in_dict, b):
    analogy_path = get_file(URL="https://raw.githubusercontent.com/nicholas-leonard/word2vec/master/questions-words.txt")
    file_path = "data/float32_analogy_vectors.npy"
    # file_path = "data/float64_analogy_vectors.npy"

    if not os.path.exists(file_path):
        file = open(analogy_path).readlines()[1:]
        file = [line for line in file if len(line.split()) == 4]
        timerD("Filtered line")
        args = zip(repeat(file), range(len(file)), repeat(in_dict))
        vectors = q_smp(get_index_vectors, args)
        np.save(file_path, np.array(vectors, dtype=np.float64), allow_pickle=True)

    vectors = np.load(file_path, allow_pickle=True)
    hi: np.ndarray = vectors[:, 1] - vectors[:, 0] + vectors[:, 2]
    currT, bT = q_mp(convert_to_tensors, hi), convert_to_tensors(b)

    # Loop that finds all of the locations

    # Should be much more efficient, but is running slower ATM
    # timerD("Finding data")
    # data = [(each, bT) for each in currT]
    # tf_layer = AnalogyLayer()
    # locs = [tf_layer(d) for d in data]

    locs = [do_it_all(each, bT) for each in currT]
    return locs



if __name__ == '__main__':
    os.system("clear")
    timerD("Importing libraries")

    words = ["car", "automobile", "truck", "bus", "limo", "jeep", "boat", "canoe", "dinghy", "motorboat", "yacht", "catamaran"]

    n = 300
    # a, b, c = load_embeddings(n=n, glove_size=840)
    a, b, c = load_embeddings(n=n, glove_size=6)
    # print(*a.tolist(),sep='\n')
    analogies = load_analogies(c, b)

    time_taken = timerD(f"Math done!", return_difference=True)
    print(f" Seconds per analogy: {time_taken / len(analogies)}")
    print(f" Analogies per second: {len(analogies) / time_taken}")
    # for each in analogies:
    #     print(a[each])
    # print(a[analogies])
    exit()
    # word_list, vector_list, word_vector_dict = a, b, c
    timerD(f"load_embeddings({n}) complete")

    bT = tf.convert_to_tensor(b)
    timerD("Loading b into a Tensor")

    idx = 0
    curr = bT[idx]
    for each in bT:
        x = tf.subtract(curr, each)
        x = tf.abs(x)
        x = tf.reduce_sum(x)
        # print(x)
        # exit()
    timerD("Found each distance")
    print(curr.shape)
    q = tf.reshape(tf.repeat(curr, len(bT)), shape=(len(bT), len(curr)))
    print(q.shape)

    x = tf.subtract(q, bT)
    x = tf.abs(x)
    x = tf.reduce_sum(x, axis=-1)
    x = x.numpy().tolist()
    x.sort()
    distances = x[:5]

    print(distances)

    # all = tf.expand_dims(list(embeddings_dict.curr()), 1)

    # timerD("Entering do_it_all")
    # do_it_all(embeddings_dict.items())
    # timerD("Completed do_it_all")

    # print(f"embeddings_dict = {embeddings_dict.keys()[keys]}")
    # print()
    # all = tf.expand_dims(list(embeddings_dict.curr()), 1)
    # for k, v in word_dict.items():
    # for k, v in embeddings_dict.items():
    # print(v, k)
    # print(f"{word}: {np.mean(embeddings_dict[word])}")
    # distance = tf.reduce_sum(tf.abs(tf.subtract(v, all)), axis=2)
    # distance = tf.reduce_sum(tf.abs(tf.subtract(v, tf.expand_dims(list(word_dict.curr()), 1))), axis=2)
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
