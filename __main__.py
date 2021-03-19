import multiprocessing as mp
# import jax as jnp
import os
from itertools import repeat

import numpy as np

from shared_functions import timerD

os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '3')
import tensorflow as tf
from tensorflow.python.layers.base import Layer

for gpu in tf.config.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)


def get_analogy_file_path():
    return get_file(URL="https://raw.githubusercontent.com/nicholas-leonard/word2vec/master/questions-words.txt")


def open_analogy_file():
    return open(get_analogy_file_path())


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

    return words, np.squeeze(vectors)


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


def prep_embeddings_mp(glove_file, proc_count, N):
    batch_size = int(len(glove_file) / N)
    lines = glove_file[proc_count * batch_size:(proc_count + 1) * batch_size]
    words = ["" for _ in range(len(lines))]
    vectors = [[0.0 for __ in range(len(lines[0].split()) - 1)] for _ in range(len(lines))]
    words, vectors = np.array(words, dtype=object), np.zeros_like(vectors, dtype=np.float32)

    return words, np.squeeze(vectors)


# Slices up the the glovefiles into separate files of words and vectors
def prepare_embeddings_files(n=300, glove_size=6):
    new_folder = f"data/glove/{glove_size}B.{n}d"
    os.system(f"mkdir -p {new_folder}")
    vector_path = f"{new_folder}/glove.{glove_size}B.{n}d.vectors"
    word_path = f"{new_folder}/glove.{glove_size}B.300d.words"
    glove_path = f"{new_folder}/glove.{glove_size}B.300d.txt"
    # word_npy_path = f"{new_folder}/glove.{glove_size}B.{n}d.words.npy"
    vector_npy_path = f"{new_folder}/glove.{glove_size}B.{n}d.vectors.npy"
    word_path_sorted = f"{new_folder}/glove.{glove_size}B.300d.words_sorted"
    vector_npy_path_sorted = f"{new_folder}/glove.{glove_size}B.{n}d.vectors_sorted.npy"

    # print(len(open(glove_path).readlines()))
    # exit()
    ###################################################################################################
    ###################################################################################################
    if not os.path.exists(vector_path) or not os.path.exists(word_path):
        fW = open(word_path, "w")
        fV = open(vector_path, "w")
        for line in open(glove_path):
            data = line.split(' ')
            print(data[0][0], end='\r')
            word, vector = ''.join(data[0]), np.array(data[1:], dtype=np.float32)
            fW.write(f"{word}\n")
            for each in vector:
                fV.write(f"{str(each)} ")
            fV.write("\n")
        timerD(f"Preparing {vector_path.rsplit('/', 1)[1]} and {word_path.rsplit('/', 1)[1]}")

    # if not os.path.exists(word_npy_path):
    #     timerD(f"Preparing {word_npy_path}")
    #     word_lines = open(word_path).readlines()
    #     word_array = ["" for _ in range(len(word_lines))]
    #     for index, line in enumerate(word_lines):
    #         print(f"{index / len(word_lines) * 100:2.1f}%", end='\r')
    #         word_array[index] = np.array(line.split(), dtype=object)[0]
    #     np.save(word_npy_path, word_array, allow_pickle=True)

    if not os.path.exists(vector_npy_path):
        vector_lines = open(vector_path).readlines()
        vector_array = [[0.0 for __ in range(300)] for _ in range(len(vector_lines))]
        for index, line in enumerate(vector_lines):
            print(f"{index / len(vector_lines) * 100:2.1f}%", end='\r')
            vector_array[index] = np.array(line.split(), dtype=np.float32)[:]
        np.save(vector_npy_path, vector_array, allow_pickle=True)
        timerD(f"Preparing {vector_npy_path}")

    if not os.path.exists(vector_npy_path_sorted) or not os.path.exists(word_path_sorted):
        a = np.array(open(word_path).read().split())
        b = np.load(vector_npy_path, allow_pickle=True)
        a, b = zip([a, b])
        a, b = np.concatenate([*a]), np.concatenate([*b])
        a, b = zip(*sorted(((zip(a, b)))))
        b = np.array(b, dtype=object)
        # np.save(vector_npy_path_sorted, b, allow_pickle=True)

        clean_b = []
        fW = open(word_path_sorted, "w")
        for index, (q, w) in enumerate(zip(a, b)):
            if len(w) == 300:
                fW.write(f"{q}\n")
                clean_b.append(b[index])

        np.save(vector_npy_path_sorted, clean_b, allow_pickle=True)
        timerD(f"Sorting and saving to {word_path_sorted.rsplit('/', 1)[1]} and {vector_npy_path_sorted.rsplit('/', 1)[1]}")


def load_embeddings(n=300, glove_size=6):
    new_folder = f"data/glove/{glove_size}B.{n}d"
    os.system(f"mkdir -p {new_folder}")
    vector_path = f"{new_folder}/glove.{glove_size}B.{n}d.vectors.npy"
    word_path = f"{new_folder}/glove.{glove_size}B.300d.words.npy"
    word_path = f"{new_folder}/glove.{glove_size}B.300d.words"
    glove_path = f"{new_folder}/glove.{glove_size}B.300d.txt"
    word_path_sorted = f"{new_folder}/glove.{glove_size}B.300d.words_sorted"
    vector_npy_path_sorted = f"{new_folder}/glove.{glove_size}B.{n}d.vectors_sorted.npy"

    ###################################################################################################
    ###################################################################################################
    if glove_size > 6:
        prepare_embeddings_files(n=n, glove_size=glove_size)
    else:
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

            fW = open(word_path, "w")
            for word in word_array:
                fW.write(f"{word}\n")

            if not os.path.exists(vector_npy_path_sorted): np.save(vector_npy_path_sorted, vector_array, allow_pickle=True)
            if not os.path.exists(word_path_sorted): np.save(word_path_sorted, word_array, allow_pickle=True)

    ###################################################################################################
    ###################################################################################################

    a = np.array(open(word_path_sorted).read().split())
    b = np.load(vector_npy_path_sorted, allow_pickle=True)

    # Create a dict with pointers to vector_list
    c = {str(a[idx]): b[idx] for idx in range(len(a))}

    word_array, vector_array, word_vector_dict = np.array(a, dtype=object), np.array(b), c
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


# Splits and gets the last element in lower case
def analogy_answer_splitter(array, index):
    return array[index].split()[-1].lower()


def mp_split_vectors(array, index, in_dict):
    each = array[index].split()
    return [in_dict[e] for e in each]


def get_vectors(array, in_dict):
    return [in_dict[e.lower()] for e in array]


def get_index_vectors(array, index, in_dict):
    return [in_dict[w.lower()] for w in array[index].split()]


def score_if_same(array, array2, index):
    return array[index] == array2[index]


def get_analogy_vector(array, index, in_dict):
    vectors = [in_dict[w.lower()] for w in array[index].split()]
    result = vectors[1] - vectors[0] + vectors[2]
    return result


class AnalogyIndiceLayer(Layer):
    def __init__(self):
        super(AnalogyIndiceLayer, self).__init__()

    @tf.function(experimental_compile=True)
    def call(self, inputs, training=False):
        # Like a TF functional model
        x = tf.subtract(inputs[0], inputs[1])
        x = tf.abs(x)
        x = tf.reduce_sum(x, axis=1)
        x = tf.argmin(x)  # Can be easily changed to grab the K-nearest
        return x


def convert_to_tensors(curr, dtype=tf.float32):
    return tf.convert_to_tensor(curr, dtype=dtype)


def calculate_analogy_vectors(in_dict, vector_list, limit=None):
    analogy_path = get_file(URL="https://raw.githubusercontent.com/nicholas-leonard/word2vec/master/questions-words.txt")
    file_path = "data/float32_analogy_vectors.npy"

    ###################################################################################################
    ###################################################################################################

    if not os.path.exists(file_path):
        file = [line for line in open(analogy_path).readlines() if len(line.split()) == 4]
        vectors = q_smp(get_index_vectors, zip(repeat(file), range(len(file)), repeat(in_dict)))
        np.save(file_path, np.array(vectors, dtype=np.float32), allow_pickle=True)

    ###################################################################################################
    ###################################################################################################

    vectors = np.load(file_path, allow_pickle=True)
    vectors = vectors[:limit]  # Limit the amount of vectors, for debugging
    predicted_vectors: np.ndarray = vectors[:, 1] - vectors[:, 0] + vectors[:, 2]
    convert_to_tensors(predicted_vectors)
    convert_to_tensors(vector_list)
    currT, bT = q_mp(convert_to_tensors, predicted_vectors), convert_to_tensors(vector_list)

    timerD("Converting data to tensors")
    tf_layer = AnalogyIndiceLayer()
    locs = list(map(lambda x: tf_layer((x, bT)).numpy(), currT))

    return locs


def get_analogy_answers():
    analogy_path = get_file(URL="https://raw.githubusercontent.com/nicholas-leonard/word2vec/master/questions-words.txt")
    file_path = "data/analogy_answers.txt"

    ###################################################################################################
    ###################################################################################################

    if not os.path.exists(file_path) or True:
        file = [line for line in open(analogy_path).readlines() if len(line.split()) == 4]
        analogy_answers = q_smp(analogy_answer_splitter, zip(repeat(file), range(len(file))))
        with open(file_path, "w") as f:
            for line in analogy_answers:
                f.write(f"{line}\n")

    ###################################################################################################
    ###################################################################################################

    analogy_answers = open(file_path).read().splitlines()

    return analogy_answers


if __name__ == '__main__':
    os.system("clear")
    timerD("Importing libraries")
    # prepare_embeddings_files(glove_size=6)
    # prepare_embeddings_files(glove_size=840)
    # exit()

    words = ["car", "automobile", "truck", "bus", "limo", "jeep", "boat", "canoe", "dinghy", "motorboat", "yacht", "catamaran"]

    n, glove_size = 300, 6
    # n, glove_size = 300, 42
    # n, glove_size = 300, 840
    limit = None  # Reduces the amount of words done, for debugging. Set to None for the full dataset.

    # n, glove_size = 300, 6
    # limit = 5

    word_list, vector_list, word_vector_dict = load_embeddings(n=n, glove_size=glove_size)
    # print(word_list)
    # exit()

    predicted_analogy_indices = calculate_analogy_vectors(word_vector_dict, vector_list, limit=limit)
    timerD(f"Predicting analogy indices")

    predicted_analogies = word_list[predicted_analogy_indices]
    analogy_answers = get_analogy_answers()
    timerD("Loading analogy answers")

    args = zip(repeat(predicted_analogies), repeat(analogy_answers), range(len(predicted_analogies)))
    true_if_correct = q_smp(score_if_same, args)
    number_correct = np.sum(true_if_correct)
    percent_correct = f"{number_correct / len(predicted_analogies) * 100:2.1f}"
    timerD(f"Checking accuracy. Results (correct/total): {number_correct} / {len(predicted_analogies)} ({percent_correct}%)")

    analogies = open_analogy_file().readlines()

    categories = []
    for idx, each in enumerate(analogies):
        if len(each.split()) != 4:
            category_name = each.split()[1]
            categories.append([category_name, idx])

    categorized = []
    for i in range(len(categories)):
        if i < len(categories) - 1:
            idx0, idx1 = categories[i][1], categories[i + 1][1]
            t0, t1 = analogy_answers[idx0:idx1], predicted_analogies[idx0:idx1]
        else:
            idx0 = categories[i][1]
            t0, t1 = analogy_answers[idx0:], predicted_analogies[idx0:]

        args = zip(repeat(t0), repeat(t1), range(len(t0)))
        number_correct = np.sum(q_smp(score_if_same, args))
        percent_correct = f"{number_correct / len(t1) * 100:02.1f}"
        category_name = categories[i][0]
        print(f"{category_name:>30}, Accuracy: {percent_correct}% ({number_correct}/{len(t1)})")

    # predictions_dir = f"data/predictions"
    # os.system(f"mkdir -p {predictions_dir}")
    # incorrect_predictions_path = f"{predictions_dir}/wrong_predictions-glove_{glove_size}-correct_{percent_correct}p.txt"
    # with open(incorrect_predictions_path, 'w') as f:
    #     for a, b, c in zip(predicted_analogies, analogy_answers, true_if_correct):
    #         if not c:
    #             pretty = f"{a:>20} | {b:<20}"
    #             f.write(f"{pretty}\n")
    #             # print(pretty)

    # timerD(f"Checking accuracy. Results (correct/total): {number_correct} / {len(predicted_analogies)} ({percent_correct}%)")

    timerD(f"Exiting...", show_total_time_elapsed=True)
