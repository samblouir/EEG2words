# with open("words_in_vocab.txt", 'w') as f:
#     collection = ' '.join(words)
#     f.write(collection)
# for word in words:
#     f.write()


# import numpy as np
#
# vectors = open("tmp/uni-vecs-685.tsv").read().splitlines()
# vectors = list(map(lambda x: [float(e) for e in x.split("\t")], vectors))
# vectors = np.asarray(vectors).astype(np.float32)
# labels = open("tmp/uni_labels-685.tsv").read().splitlines()
# vector_label_dict = {k:v for k,v in zip(labels,vectors)}
#
# for k,v in vector_label_dict.items():
#     print(k,v.shape)
import datetime

import train3_proc

with open("words_in_vocab.txt", 'r') as f:
    words = f.read().split()

import pywt
# import importlib
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import numpy as np
# import pywt as wt
# import pickle
import io
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# torch.manual_seed(42)
import multiprocessing as mp
import os, pickle

from shared_functions import timerD

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # disable the GPU

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds

if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')

    try:
        if gpus:
            for gpu in gpus:
                pass
                tf.config.experimental.set_memory_growth(gpu, True)
    except:
        pass

    tf.config.experimental.enable_tensor_float_32_execution(True)

dim = 50

print("unprotected print!")

## Paths
trainfile = "D:\\NLP Datasets\\train_dict.dict"
devfile = "D:\\NLP Datasets\\dev_dict.dict"
testfile = "D:\\NLP Datasets\\test_dict.dict"
new_train_data_path = "D:\\NLP Datasets\\test_dict.dict"
wavelet_trainfile = "D:\\NLP Datasets\\wavelet_train_dict.dict"
wavelet_testfile = "D:\\NLP Datasets\\wavelet_test_dict.dict"
wavelet_devfile = "D:\\NLP Datasets\\wavelet_dev_dict.dict"

train_dataset_path = "D:\\NLP Datasets\\train_dataset.tensorflow"
train_dataset_elementspec_path = "D:\\NLP Datasets\\elementspec_train_dataset.tensorflow"
train_dataset_vectorized_dict = "D:\\NLP Datasets\\elementspec_train_vectorized_dict.tensorflow"
train_dataset_vectorized_dict_R = "D:\\NLP Datasets\\elementspec_train_vectorized_dict_R.tensorflow"
test_dataset_vectorized_dict = "D:\\NLP Datasets\\elementspec_test_vectorized_dict.tensorflow"
test_dataset_vectorized_dict_R = "D:\\NLP Datasets\\elementspec_test_vectorized_dict_R.tensorflow"
test_dataset_path = "D:\\NLP Datasets\\train_dataset.tensorflow"
test_dataset_elementspec_path = "D:\\NLP Datasets\\elementspec_train_dataset.tensorflow"

if not os.path.exists(test_dataset_path):
    os.system("mkdir -p data/WD/test_dataset")
    test_dataset_path = "data/WD/test_dataset/test_dataset.tensorflow"
if not os.path.exists(test_dataset_elementspec_path):
    test_dataset_elementspec_path = "data/WD/test_dataset/test_dataset_elementspec"
if not os.path.exists(train_dataset_vectorized_dict):
    train_dataset_vectorized_dict = "data/WD/train_dataset/elementspec_train_vectorized_dict.tensorflow"
if not os.path.exists(test_dataset_vectorized_dict):
    test_dataset_vectorized_dict = "data/WD/test_dataset/elementspec_test_vectorized_dict.tensorflow"
if not os.path.exists(train_dataset_vectorized_dict_R):
    train_dataset_vectorized_dict = "data/WD/train_dataset/elementspec_train_vectorized_dict_R.tensorflow"
if not os.path.exists(test_dataset_vectorized_dict_R):
    test_dataset_vectorized_dict_R = "data/WD/test_dataset/elementspec_test_vectorized_dict_R.tensorflow"

if not os.path.exists(train_dataset_path):
    os.system("mkdir -p data/WD/train_dataset")
    train_dataset_path = "data/WD/train_dataset/train_dataset.tensorflow"
if not os.path.exists(train_dataset_elementspec_path):
    train_dataset_elementspec_path = "data/WD/train_dataset/train_dataset_elementspec"

save_weight_path = "D:\\NLP Datasets\\TFModel_encoder.weights"
save_weight_path_backup = "D:\\NLP Datasets\\TFModel_encoder_backup.weights"
save_weight_path_last_epoch = "D:\\NLP Datasets\\TFModel_encoder_last_epoch.weights"
save_weight_path_all_done = "D:\\NLP Datasets\\TFModel_encoder_all_epochs_done.weights"

if not os.path.exists(save_weight_path):
    save_weight_path = "TFModel_encoder.weights"
if not os.path.exists(save_weight_path_backup):
    save_weight_path_backup = "TFModel_encoder_backup.weights"
if not os.path.exists(save_weight_path_last_epoch):
    save_weight_path_last_epoch = "TFModel_encoder_last_epoch.weights"
if not os.path.exists(save_weight_path_all_done):
    save_weight_path_last_epoch = "TFModel_encoder_all_epochs_done.weights"

if not os.path.exists(trainfile):
    trainfile = "data/WD/train_dict.dict"
if not os.path.exists(devfile):
    devfile = "data/WD/dev_dict.dict"
if not os.path.exists(testfile):
    testfile = "data/WD/test_dict.dict"
if not os.path.exists(new_train_data_path):
    new_train_data_path = "data/WD/new_train_data.npy"
    # new_train_data_path = "data/WD/new_train_data_mini.npy"
if not os.path.exists(wavelet_trainfile):
    wavelet_trainfile = "data/WD/wavelet_train_dict.dict"
if not os.path.exists(wavelet_testfile):
    wavelet_testfile = "data/WD/wavelet_test_dict.dict"
if not os.path.exists(wavelet_devfile):
    wavelet_devfile = "data/WD/wavelet_dev_dict.dict"


def quick_save(in_file, out_path):
    pickle.dump(in_file, open(out_path, 'wb'))
    print(f"  Saved to {out_path}!")


def quick_load(in_path):
    return pickle.load(open(in_path, 'rb'))


class a_tf_model(tf.keras.Model):
    def __init__(self, *args, **kwargs):
        super(a_tf_model, self).__init__(*args, **kwargs)

        # supported_kwargs = ['inputs', 'outputs', 'name', 'trainable', 'skip_init']
        # model_kwargs = {k: kwargs[k] for k in kwargs if k in supported_kwargs}
        #
        # other_kwargs = {k: kwargs[k] for k in kwargs if k not in supported_kwargs}
        # a = model_kwargs['inputs']
        # b = model_kwargs['outputs']
        # self.model = tf.keras.models.Model(a,b)


def load_model():
    dropout = 0.5
    # inputs = tf.keras.layers.Input(shape=(channel_indxs, 28))
    inputs = tf.keras.layers.Input(shape=(51, 105))
    inputs = tf.keras.layers.Input(shape=(2, 105, 28))
    # inputs = tf.keras.layers.Input(shape=(105, 28))
    nh, kd = 3, 3
    A = tf.keras.layers.MultiHeadAttention(num_heads=nh, key_dim=kd, dropout=0.1)
    B = tf.keras.layers.MultiHeadAttention(num_heads=nh, key_dim=kd, dropout=0.5)
    C = tf.keras.layers.MultiHeadAttention(num_heads=nh, key_dim=kd, dropout=0.5)
    D = tf.keras.layers.MultiHeadAttention(num_heads=nh, key_dim=kd, dropout=0.5)
    x = inputs
    # x = tf.keras.layers.Reshape(target_shape=(2, 105, 28))(x)

    x = tf.keras.layers.Conv1D(filters=16, kernel_size=2, strides=2, padding='same', )(x)
    x = tfa.layers.FilterResponseNormalization()(x)
    x = tfa.layers.TLU()(x)
    x = tf.keras.layers.SpatialDropout2D(dropout)(x)
    #
    x = D(x, x, return_attention_scores=False)
    # x = tf.keras.layers.SpatialDropout2D(0.25)(x)
    x = tfa.layers.FilterResponseNormalization()(x)
    x = tfa.layers.TLU()(x)
    x = tf.keras.layers.SpatialDropout2D(dropout)(x)

    # x = tf.keras.layers.Conv2D(filters=64, kernel_size=2, strides=2, padding='same', )(x)
    # x = tfa.layers.FilterResponseNormalization()(x)
    # x = tfa.layers.TLU()(x)

    x = C(x, x, return_attention_scores=False)
    x = tfa.layers.FilterResponseNormalization()(x)
    x = tfa.layers.TLU()(x)
    x = tf.keras.layers.SpatialDropout2D(dropout)(x)

    x = tf.keras.layers.Conv1D(filters=16, kernel_size=3, strides=3, padding='same', )(x)
    x = tfa.layers.FilterResponseNormalization()(x)
    # x = tf.keras.layers.BatchNormalization()(x)
    x = tfa.layers.TLU()(x)
    x = tf.keras.layers.SpatialDropout2D(dropout)(x)

    x = tf.keras.layers.Flatten()(x)
    # x = tfa.layers.NoisyDense(214)(x)
    x = tfa.layers.NoisyDense(144)(x)

    x = tf.keras.layers.Dropout(dropout)(x)
    x = tfa.layers.NoisyDense(100)(x)
    # x = tf.keras.layers.Dense(1696)(x)
    # x = tf.keras.layers.Dense(800)(x)
    # x = tf.keras.layers.Dense(214)(x)
    # x = tf.keras.layers.Dense(144)(x)
    # x = tf.keras.layers.Dense(100)(x)
    # x = tf.keras.layers.Dense(2)(x)
    x = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)  # L2 normalize embeddings
    outputs = x

    model = tf.keras.models.Model(inputs, outputs)
    # model = a_tf_model(inputs=inputs, outputs=outputs)
    # tf.keras.layers.Dense(100, activation=None),  # No activation on final dense layer

    LR_mult = 1
    LR = tfa.optimizers.ExponentialCyclicalLearningRate(
        initial_learning_rate=1e-4 * LR_mult,
        maximal_learning_rate=1e-2 * LR_mult,
        step_size=2000,
        # initial_learning_rate=1e-2,
        # maximal_learning_rate=1.0,
        # step_size=2000,
        gamma=0.96,
    )

    optimizer = tfa.optimizers.LAMB(learning_rate=LR)
    # LR = tfa.optimizers.CyclicalLearningRate(1e-3, 1e-1, step_size=2000, scale_fn=lambda x: 0.96)
    loss = tfa.losses.TripletSemiHardLoss()
    # loss = tfa.losses.TripletHardLoss()
    model.compile(
        optimizer=optimizer,
        loss=loss,
    )
    try:
        # model.load_weights(save_weight_path)
        os.system("clear")
        # model.load_weights(save_weight_path, by_name=True, skip_mismatch=True)
    except:
        try:
            # model.load_weights(save_weight_path)
            pass
        except:
            print("  Oddities while loading the save.")
    # return model, loss
    return model


if __name__ == "__main__":
    # temp = load_model()
    # model, model_loss = temp[0], temp[1]
    model = load_model()
    model_loss = tfa.losses.TripletHardLoss()
    print(model)
    print(model_loss)
    # exit()


class callback3(tf.keras.callbacks.Callback):
    def __init__(self, testfile="data/WD/test_dict.dict", in_x=None, in_y=None, y_dict_count=None):
        super(callback3, self).__init__()
        # self.best_loss = 99999
        # self.best_loss_tsv_save = 99999
        self.x = in_x
        self.y = in_y
        self.y_dict_count = y_dict_count
        self.can_save = False

    # def on_epoch_begin(self, epoch, logs=None):
    #     global model_loss, monkey, monkey_dict, y_dict
    #     results = self.model.predict(self.x)
    #     # new_res = tfa.losses.triplet_hard_loss(self.y, results, distance_metric="L2")  # (results, self.y)
    #     # print(new_res)
    #     counter = 0
    #     monkey_weights = []
    #     for k, size in self.y_dict_count.items():
    #         med_dist = tf.math.reduce_euclidean_norm(results[counter:counter + size])
    #         monkey_weights.append(med_dist)
    #         # print(med_dist, counter, size)
    #         counter += size
    #         # monkey = counter
    #
    #     monkey_weights = tf.divide(monkey_weights, tf.reduce_max(monkey_weights))
    #     # monkey_weights = tf.add(monkey_weights, 1)
    #     # monkey_weights = tf.square(monkey_weights, 2)
    #     # monkey_weights = tf.divide(monkey_weights, tf.reduce_max(monkey_weights))
    #     # monkey_weights = tf.add(monkey_weights, -1)
    #     # monkey_weights = tf.square(monkey_weights, 2)
    #
    #     mean_mw = tf.reduce_mean(monkey_weights)
    #     print(f"  mean_mw: {mean_mw}")
    #
    #     for k, mw in zip(y_dict.keys(), monkey_weights):
    #         monkey_dict[k] = mw

    # for k, v in monkey_dict.items():
    #     print(k, v)
    # exit()

    def on_epoch_end(self, epoch, logs=None):
        global testdata
        global best_loss, best_loss_tsv_save
        global epoch_count
        # timerD("", silent=True)
        print(f"  Epoch #{epoch_count}:  ", end='')
        if logs is not None:
            for k, v in logs.items():
                print(f"  {k}: {v:4f}", end=',  ')
        print(f'best loss: {best_loss:4f}', end=', ')
        print(f'best loss saved: {best_loss_tsv_save:4f}', end=', ')

        if os.path.exists('tmp/force_save'):
            os.system("rm tmp/force_save")
            best_loss_tsv_save = 99999
            best_loss = 99999
            self.can_save = True

        # self.model.save_weights(save_weight_path_last_epoch)

        if 'val_loss' in logs:
            loss_to_save_on = logs['val_loss']
        else:
            loss_to_save_on = logs['loss']

        if loss_to_save_on <= best_loss or self.can_save:

            # if loss_to_save_on < self.best_loss_tsv_save * 0.9 or can_save:
            if self.can_save:
                self.model.save_weights("tmp/temp_model_weights")
                self.model.load_weights(save_weight_path)

                # From: https://www.tensorflow.org/addons/tutorials/losses_triplet
                results = self.model.predict(self.x)
                save_path = f"tmp/vecs-{len(self.x)}.tsv"
                np.savetxt(save_path, results, delimiter='\t')

                all_avgs = []
                avgs = []
                last_y = self.y[0]
                for idx, (x, y) in enumerate(zip(results, self.y)):
                    if last_y != y:
                        new_avg = np.mean(avgs, axis=0)
                        all_avgs.append(new_avg)
                        avgs.clear()
                        last_y = y

                    avgs.append(x)
                    if idx == len(self.y) - 1 or idx == len(results) - 1:
                        new_avg = np.mean(avgs, axis=0)
                        all_avgs.append(new_avg)

                results = np.asarray(all_avgs).astype(np.float32)
                save_path = f"tmp/uni-vecs-{len(self.x)}.tsv"
                np.savetxt(save_path, results, delimiter='\t')
                print(f"    New best loss found! {loss_to_save_on}, saved to \"{save_path}\"!")
                best_loss_tsv_save = loss_to_save_on
                self.model.load_weights("tmp/temp_model_weights")

            best_loss = loss_to_save_on
            self.model.save_weights(save_weight_path)

        # timerD(f"Finished epoch {epoch}", end='  ')
        print()


if __name__ == "__main__":
    import TFModel

    # os.system(f"rm {wavelet_trainfile}")
    if not os.path.exists(wavelet_trainfile):
        train = np.load(trainfile, allow_pickle=True, fix_imports=False)
        train = {e[0]: e[1] for e in sorted([[k, v] for k, v in train.items()])}
        TFModel.create_wavelet_dict(train, wavelet_trainfile)
        del train

    if not os.path.exists(wavelet_testfile):
        test = np.load(testfile, allow_pickle=True, fix_imports=False)
        test = {e[0]: e[1] for e in sorted([[k, v] for k, v in test.items()])}
        TFModel.create_wavelet_dict(test, wavelet_testfile)
        del test

    if not os.path.exists(wavelet_devfile):
        dev = np.load(devfile, allow_pickle=True, fix_imports=False)
        dev = {e[0]: e[1] for e in sorted([[k, v] for k, v in dev.items()])}
        TFModel.create_wavelet_dict(dev, wavelet_devfile)
        del dev

    del TFModel


def create_train_dataset():
    # if not os.path.exists(train_dataset_path):

    if not os.path.exists(train_dataset_path) or not os.path.exists('tmp/vector_to_key_dict'):
        print("Creating new train dataset...")
        os.system("mkdir -p tmp")

        with open(wavelet_trainfile, 'rb') as datafile:
            wavelet_trainfile_data = pickle.load(datafile)

        data_to_use = wavelet_trainfile_data

        # Vectorize the keys
        key_to_vector_dict = {}
        vector_to_key_dict = {}
        for idx0, (k, v) in enumerate(data_to_use.items()):
            if k not in key_to_vector_dict:
                val = len(key_to_vector_dict)
                key_to_vector_dict[k] = len(key_to_vector_dict)
                vector_to_key_dict[val] = k

        pickle.dump(key_to_vector_dict, open(train_dataset_vectorized_dict, 'wb'))
        pickle.dump(vector_to_key_dict, open(train_dataset_vectorized_dict_R, 'wb'))

        x, y = [], []
        for idx0, (k, v) in enumerate(data_to_use.items()):
            for idx1, each in enumerate(v):
                x.append(each)
                y.append(key_to_vector_dict[k])

        train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
        tf.data.experimental.save(train_dataset, train_dataset_path)
        pickle.dump(train_dataset.element_spec, open(train_dataset_elementspec_path, 'wb'))
        pickle.dump(vector_to_key_dict, open('tmp/vector_to_key_dict', 'wb'))
        timerD("Saved the train dataset to the disk!")


def create_test_dataset():
    if not os.path.exists(test_dataset_path):
        with open(wavelet_testfile, 'rb') as datafile:
            wavelet_testfile_data = pickle.load(datafile)

        data_to_use = wavelet_testfile_data

        # Vectorize the keys
        # key_to_vector_dict = {}
        # vector_to_key_dict = {}

        with open(train_dataset_vectorized_dict, 'rb') as datafile:
            key_to_vector_dict = pickle.load(datafile)

        with open(train_dataset_vectorized_dict_R, 'rb') as datafile:
            vector_to_key_dict = pickle.load(datafile)

        for idx0, (k, v) in enumerate(data_to_use.items()):
            if k not in key_to_vector_dict:
                val = len(key_to_vector_dict)
                key_to_vector_dict[k] = len(key_to_vector_dict)
                vector_to_key_dict[val] = k

        # quick_save(vector_to_key_dict,'test_vector_to_key_dict')

        pickle.dump(key_to_vector_dict, open(test_dataset_vectorized_dict, 'wb'))
        pickle.dump(vector_to_key_dict, open(test_dataset_vectorized_dict_R, 'wb'))

        x, y = [], []
        for idx0, (k, v) in enumerate(data_to_use.items()):
            for idx1, each in enumerate(v):
                x.append(each)
                y.append(key_to_vector_dict[k])

        test_dataset = tf.data.Dataset.from_tensor_slices((x, y))
        tf.data.experimental.save(test_dataset, test_dataset_path)
        # (TensorSpec(shape=(2, 105, 28), dtype=tf.float32, name=None), TensorSpec(shape=(), dtype=tf.int32, name=None))
        pickle.dump(test_dataset.element_spec, open(test_dataset_elementspec_path, 'wb'))
        timerD("Saved the test dataset to the disk!")


if __name__ == "__main__":
    model = load_model()


def choppy(idx, selection):
    print(idx, flush=True)
    two = selection[idx]
    twosies = []
    for hundo5 in two:
        hundies = []
        for twn8 in hundo5:
            hundies.append(twn8)
        twosies.append(hundies)
    return twosies


def choppyEZ(two):
    twosies = []
    for hundo5 in two:
        hundies = []
        for twn8 in hundo5:
            hundies.append(twn8)
        twosies.append(hundies)
    return twosies


def new_plan():
    # global model
    # os.system("clear")

    timerD(f"  Finished with imports")

    ## Prepare the train dataset
    timerD("Loading the train dataset from the disk...")
    create_train_dataset()
    trainspec = quick_load(train_dataset_elementspec_path)
    train_dataset = tf.data.experimental.load(train_dataset_path, trainspec)

    create_test_dataset()
    vector_to_key_dict = quick_load('tmp/vector_to_key_dict')

    if not os.path.exists("xL") or not os.path.exists("yL"):
        timerD("Saving new xL and yL to the disk...")
        xL = np.zeros(shape=len(train_dataset), dtype=object)
        yL = np.zeros(shape=len(train_dataset), dtype=object)
        for idx, (x, y) in enumerate(train_dataset):
            xL[idx] = x.numpy()
            yL[idx] = y.numpy()
        quick_save(xL, "xL")
        quick_save(yL, "yL")

    testspec = quick_load(test_dataset_elementspec_path)
    test_dataset = tf.data.experimental.load(test_dataset_path, testspec)
    if not os.path.exists("testxL") or not os.path.exists("testyL"):
        timerD("Saving new testxL and testyL to the disk...")
        testxL = np.zeros(shape=len(test_dataset), dtype=object)
        testyL = np.zeros(shape=len(test_dataset), dtype=object)
        for idx, (x, y) in enumerate(test_dataset):
            testxL[idx] = x.numpy()
            testyL[idx] = y.numpy()
        quick_save(testxL, "testxL")
        quick_save(testyL, "testyL")

    timerD("Loading testxL and testyL...")

    def pT(input, spaces=2):
        print(' ' * spaces, f"type: {type(input)}")
        try:
            for each in input:
                pT(each, spaces + 2)
                break
        except:
            pass

    # pT(yL)

    c = 0

    def load_xL(path="xL", save_path="new_xL", force_load=True):
        timerD("load_xL()...")
        xL = quick_load(path)
        if not os.path.exists(save_path) or force_load:
            new_xL = [c for c in [d for d in [e for e in xL]]]
            new_xL = np.asarray(new_xL).astype('float32')
            # quick_save(new_xL, save_path)

        # new_xL = quick_load(save_path)
        return new_xL

    def load_yL(load_path="yL", in_type=float):
        timerD("load_yL()...")
        yL = quick_load(load_path)
        yL = np.asarray(yL).astype('float32')
        return yL

    xL = load_xL()  # (342143, 2, 105, 28)
    yL = load_yL()
    xL = xL[:-1]
    yL = yL[:-1]
    xL = xL[:342138]
    yL = yL[:342138]
    xL = xL[:331776]
    yL = yL[:331776]  # train batch size
    print(f"  len(xL): {len(xL)}")
    print(f"  len(yL): {len(yL)}")

    test_xL = load_xL("testxL", "new_testxL")
    test_yL = load_yL("testyL")
    # timerD("loading new_xL...")
    # new_xL = quick_load("new_xL")

    # indices = [int(e) for e in list(np.arange(0, len(xL) - 1, 10000))]
    # indices = [int(e) for e in list(range(0, len(xL) - 1, 1000))]
    # xL = xL[indices]
    # yL = yL[indices]
    # xL = xL[:6000]
    # yL = yL[:6000]
    # xL = xL[:12000]
    # yL = yL[:12000]
    # xL = xL[:-1]
    # yL = yL[:-1]

    # xL = xL[:-1]
    # yL = yL[:-1]

    # test_range = list(range(0,len(xL),3))
    # train_range = [e for e in list(range(len(xL))) if e not in test_range]
    # test_xL = xL[test_range] # 102602
    # test_yL = yL[test_range] # 102602

    # xL = xL[train_range]
    # yL = yL[train_range]

    # from sklearn.model_selection import train_test_split
    # xL, test_xL, yL, test_yL = train_test_split(xL, yL, test_size=0.001)

    global best_loss, best_loss_tsv_save
    best_loss = 99999
    best_loss_tsv_save = 99999

    test_y_dict_count = {}
    for each in test_yL:
        key = each
        test_y_dict_count[key] = test_y_dict_count.get(key, 0) + 1

    global y_dict
    y_dict = {}
    y_dict_count = {}
    for each in yL:
        key = each
        y_dict[key] = y_dict.get(key, 0)
        y_dict[key] += 1
        y_dict_count[key] = y_dict_count.get(key, 0) + 1

    max_val = float(np.max(list(y_dict.values())))
    for k, v in y_dict.items():
        y_dict[k] /= max_val

    total = float(np.sum(list(y_dict.values())))
    for k, v in y_dict.items():
        y_dict[k] /= total
        y_dict[k] = 1 / y_dict[k]

    max_val = float(np.max(list(y_dict.values())))
    for k, v in y_dict.items():
        y_dict[k] /= max_val

    for k, v in y_dict.items():
        y_dict[k] = float(v)

    print()
    for idx, (k, v) in enumerate(y_dict.items()):
        print(f"  {k}: {v}")
        if idx == 5:
            break

    timerD("Created class weights!")

    label_path = f'tmp/labels-{len(yL)}.tsv'
    if not os.path.exists(label_path):
        with open(label_path, 'w') as file:
            for y in yL:
                curr = vector_to_key_dict[y]
                print(curr, file=file)
        timerD(f"Created {label_path}!")

    uni_label_path = f'tmp/uni_labels-{len(yL)}.tsv'
    if not os.path.exists(uni_label_path):
        with open(uni_label_path, 'w') as file:
            for y in list(set(yL)):
                curr = vector_to_key_dict[y]
                print(curr, file=file)
        timerD(f"Created {uni_label_path}!")

    ## Prepare the test dataset
    create_test_dataset()
    with open(test_dataset_elementspec_path, 'rb') as datafile:
        testspec = pickle.load(datafile)
    test_dataset = tf.data.experimental.load(test_dataset_path, testspec)
    # test_dataset = test_dataset.batch(7500)
    timerD("Loaded the test dataset from the disk!")

    summarize_model = False
    if summarize_model:
        # model.build(input_shape=(None, *np.array(x[0]).shape))
        history = model.fit(
            # train_dataset,
            xL, yL,
            verbose=1,
            # steps_per_epoch=1,
            epochs=1)
        # history = model.fit(
        #     train_dataset, steps_per_epoch=1,
        #     epochs=5)
        # model.summary()
        # try:
        #     model.load_weights(save_weight_path)
        # except:
        #     print(f"  failed to load weights at {save_weight_path}")

        timerD(f"  Ready for the main fit...")
        # import tensorflow_datasets as tfds
        # train_dataset, test_dataset = tfds.load(name="mnist", split=['train', 'test'], as_supervised=True)

        # print(f"  x[0].shape: {np.array(x[0]).shape}")
        # print(f"  y[0].shape: {np.array(y[0]).shape}")
        # print(f"  x[0]: {x[0]}")
        # print(f"  y[0]: {y[0]}")
        # print("\n" * 4)

    do_training = False
    do_training = True

    # for idx, x in enumerate(xL):
    #     if idx == len(xL) - 1:
    #         print(f" {idx}  {x.shape}")
    # for idx, y in enumerate(yL):
    #     if idx == len(yL) - 1:
    #         print(f" {idx}  {y}")

    if do_training:
        # os.system("mkdir -p WD/tb")
        os.system("mkdir -p logs")
        logdir = "logs/scalars/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
        # file_writer = tf.summary.create_file_writer(logdir + "/metrics")
        # file_writer.set_as_default()

        # try:
        #     model.load_weights(save_weight_path)
        # except:
        #     print(f"  failed to load weights at {save_weight_path}")
        timerD("Fitting the model...")
        model.summary()

        global epoch_count
        for epoch_count in range(10000):
            # for _i_xL in range(len(xL)-3):
            #     curr = xL[_i_xL:_i_xL+2]
            #     labs = yL[_i_xL:_i_xL+2]
            #     pred = model(curr)
            #     loss = model_loss(labs, pred)
            #     # print(pred)
            #     print(loss)
            #     print()
            # exit()
            # xL=tf.cast(xL,dtype=tf.bfloat16)
            # yL=tf.cast(yL,dtype=tf.float32)
            global monkey_dict
            monkey_dict = y_dict
            # loss_fn = model_loss
            # 
            # LR = tfa.optimizers.ExponentialCyclicalLearningRate(
            #     # initial_learning_rate=1e-4,
            #     # maximal_learning_rate=1e-2,
            #     # step_size=2000,
            #     initial_learning_rate=1e-4,
            #     maximal_learning_rate=1e-2,
            #     step_size=2000,
            #     gamma=0.96,
            # )
            # optimizer = tfa.optimizers.LAMB(learning_rate=LR)

            # shl = tfa.losses.TripletSemiHardLoss()
            # hl = tfa.losses.TripletHardLoss()
            #
            # @tf.function()
            # def train_step(x, y, w=None):
            #
            #     with tf.GradientTape() as tape:
            #         preds = model(x, training=True)
            #         # semi_hard_loss = shl(y, preds)
            #         semi_hard_loss = model.compiled_loss(y, preds, sample_weight=w)
            #         # semi_hard_loss = shl(y, preds, sample_weights=w)
            #
            #     grads = tape.gradient(semi_hard_loss, model.trainable_weights)
            #     optimizer.apply_gradients(zip(grads, model.trainable_weights))
            #
            #     # with tf.GradientTape() as tape:
            #     #     preds = model(x, training=True)
            #     #     # hard_loss = hl(y, preds, sample_weight = w)
            #     #     hard_loss = hl(y, preds, sample_weight=w)
            #
            #     # grads = tape.gradient(hard_loss, model.trainable_weights)
            #     # optimizer.apply_gradients(zip(grads, model.trainable_weights))
            #
            #     # return hard_loss, tf.reduce_mean(w)
            #     # loss_value = (semi_hard_loss + hard_loss)
            #
            #     # return loss_value, semi_hard_loss, hard_loss, tf.reduce_sum(w)
            #     return semi_hard_loss, tf.reduce_sum(w)
            #
            #     # return {
            #     #     'loss': loss_value,
            #     #     'semi_hard_loss': semi_hard_loss,
            #     #     'hard_loss': hard_loss,
            #     # }

            # epoch_metrics = {}
            epochs = 50000

            dtype = tf.bfloat16
            np_dtype = np.float16

            dtype = tf.float32
            np_dtype = np.float32

            timerD("Casting...")
            # t_testxL = load_cast_save('ttmp/t_testxL', test_xL)
            t_testxL = tf.cast(test_xL, dtype=dtype)
            # timerD("Casted t_testxL!")
            t_testyL = tf.cast(test_yL, dtype=dtype)
            # timerD("Casted t_testyL!")
            txL = tf.cast(xL, dtype=dtype)
            # timerD("Casted txL!")
            tyL = tf.cast(yL, dtype=dtype)
            # timerD("Casted tyL!")
            tw = tf.ones(shape=len(txL), dtype=dtype)
            timerD("Casted!")

            # print(x_groups)

            # metrics = [[train_step(_x, _y) for _x, _y in zip(x, y)] for _ in range(100)]
            batch_size = 4096 * 3
            # train_batch_size = 986
            # train_batch_size = 493
            # train_batch_size = 347
            # train_batch_size = 58

            train_batch_size = 2694
            # train_batch_size = 1347
            # train_batch_size = 898

            # train_batch_size = 2304
            # train_batch_size = 5184
            # train_batch_size = 20736
            # train_batch_size = 2592
            # train_batch_size = 10368
            # train_batch_size = 256
            # train_batch_size = 32

            # train_batch_size = 13824
            # train_batch_size = 18432
            print(f"  train_batch_size: {train_batch_size}")

            weight_values = np.ones(len(txL), dtype=np_dtype)

            run_test_metric = True
            run_train_metric = True
            train_batch_size = 1024
            inner_epochs = 2

            # run_test_metric = False
            # run_train_metric = False
            # train_batch_size = 12288
            # train_batch_size = 10368
            train_batch_size = 1024
            inner_epochs = 2

            saves = [f"saves/{x}" for x in os.listdir("saves") if x[-3:] == '.h5']
            saves.sort()
            save_path = saves[0]
            timerD(f"Loading {save_path}...")
            model.load_weights(save_path)
            timerD("Loaded weights!")

            best_val_loss = float(save_path[-3:])
            timerD(f"best_val_loss: {best_val_loss}")
            
            # exit()
            # export_preds = True
            export_preds = False
            def export_preds():
                os.system("mkdir -p out")

                # it_val = 0
                # it_vals = []
                # for index, val in enumerate(list(test_y_dict_count.values())):
                #     c0 = it_val
                #     c1 = it_val + val
                #     it_val += val
                #     it_vals.append((c0, c1))
                vector_to_key_dict = quick_load('test_vector_to_key_dict')
                label_path = f'out/labels-{len(t_testyL)}.tsv'
                uni_label_path = f'out/uni-labels-{len(t_testyL)}.tsv'
                uni_labels = []
                if not os.path.exists(label_path) or not os.path.exists(uni_label_path):
                    with open(label_path, 'w') as file:
                        for y in yL:
                            curr = vector_to_key_dict[y]
                            uni_labels.append(curr)
                            print(curr, file=file)
                    timerD(f"Created {label_path}!")

                    uni_labels = sorted(set(uni_labels))
                    with open(uni_label_path, 'w') as file:
                        for uni_label in uni_labels:
                            print(uni_label, file=file)
                    timerD(f"Created {uni_label_path}!")

                # preds = model.predict(t_testxL, batch_size=batch_size)

                results = model.predict(t_testxL)
                save_path = f"out/vecs-{len(t_testxL)}.tsv"
                np.savetxt(save_path, results, delimiter='\t')

                all_avgs = []
                avgs = []
                last_y = t_testyL[0]
                for idx, (x, y) in enumerate(zip(results, t_testyL)):
                    if last_y != y:
                        new_avg = np.mean(avgs, axis=0)
                        all_avgs.append(new_avg)
                        avgs.clear()
                        last_y = y

                    avgs.append(x)
                    if idx == len(t_testyL) - 1 or idx == len(results) - 1:
                        new_avg = np.mean(avgs, axis=0)
                        all_avgs.append(new_avg)

                results = np.asarray(all_avgs).astype(np.float32)
                save_path = f"out/uni-vecs-{len(t_testxL)}.tsv"
                np.savetxt(save_path, results, delimiter='\t')

                exit()

            
            print("Starting training...", flush=True)
            for epoch in range(epochs):
                print(f"  Epoch #{epoch:3d}")

                # ##################### test stats
                if run_test_metric:
                    # @tf.function
                    def run_the_test_metric():
                        print(f"        in run_the_test_metric()!")
                        with tf.device('/device:GPU:0'):
                            # with tf.device('/device:CPU:0'):
                            # if True:
                            test_epoch_dist_means = []
                            it_val = 0
                            preds = model.predict(t_testxL, batch_size=batch_size)
                            it_vals = []
                            # our_slice = []
                            for index, val in enumerate(list(test_y_dict_count.values())):
                                c0 = it_val
                                c1 = it_val + val
                                it_val += val
                                it_vals.append((c0, c1))
                                # our_slice.append(t_testxL[c0:c1])
                            # our_slice = [model(t_testxL[c0:c1]) for c0,c1 in it_vals]
                            #
                            # Find average distance to centers
                            # for index, val in enumerate(list(test_y_dict_count.values())):
                            for c0, c1 in it_vals:
                                curr_preds = preds[c0:c1]
                                mean_loc = tf.reduce_mean(curr_preds, axis=0)
                                mega_loc = tf.expand_dims(mean_loc, axis=0)

                                each_vectors_dist_from_group_avg = tf.math.reduce_euclidean_norm(curr_preds - mega_loc, axis=1)

                                # curr_mean = tf.reduce_mean(each_vectors_dist_from_group_avg)
                                # epoch_dist_means.append(curr_mean)

                                each_vectors_dist_from_group_avg = tf.reduce_mean(each_vectors_dist_from_group_avg)
                                test_epoch_dist_means.append(each_vectors_dist_from_group_avg)
                            # print('running MP')
                            # from itertools import repeat
                            # # args = zip(repeat(preds), repeat(it_vals), range(len(it_vals)))
                            # preds=[]
                            # args = zip(repeat(preds), repeat(it_vals), range(len(it_vals)), our_slice)
                            # print("starting!")
                            # test_epoch_dist_means = mp.Pool(mp.cpu_count()).starmap(train3_proc.mp_cruncher, args)
                            #     # mp_cruncher(preds, it_vals, index)

                            with tf.summary.create_file_writer(logdir + "/validation").as_default(epoch):
                                tf.summary.scalar("mean class dist to centers", data=np.mean(test_epoch_dist_means), step=epoch)
                            print(f"    test mean class dist to centers: {np.mean(test_epoch_dist_means)}")
                            test_result = np.mean(test_epoch_dist_means)
                            return test_result
                            # del  preds, each_vectors_dist_from_group_avg, curr_preds, mega_loc, mean_loc

                    test_result = run_the_test_metric()
                    curr_val_loss = test_result
                    if curr_val_loss < best_val_loss:
                        best_val_loss = curr_val_loss
                        # save_dir = f"saves/{curr_val_loss}/"
                        save_dir = f"saves/"
                        os.system(f'mkdir -p {save_dir}')
                        model.save_weights(f"{save_dir}/{curr_val_loss}.h5")
                        model.save_weights(f"saves/best_save.h5")
                        print(f" *** new best val loss of {best_val_loss}!")
                        export_preds()

                    # timerD("Ran test metric!")

                ################
                all_metrics = []
                if run_train_metric:
                    def run_the_train_metric():
                        # with tf.device('/device:CPU:0'):
                        with tf.device('/device:GPU:0'):
                            # Find average distance to centers
                            epoch_dist_means = []
                            weight_values = []
                            preds = model.predict(txL, batch_size=batch_size)
                            it_val = 0
                            for index, val in enumerate(list(y_dict_count.values())):
                                c0 = it_val
                                c1 = it_val + val
                                it_val += val
                                curr_preds = preds[c0:c1]
                                mean_loc = tf.reduce_mean(curr_preds, axis=0)
                                mega_loc = tf.expand_dims(mean_loc, axis=0)

                                each_vectors_dist_from_group_avg = tf.math.reduce_euclidean_norm(curr_preds - mega_loc, axis=1)

                                curr_mean = tf.reduce_mean(each_vectors_dist_from_group_avg)
                                epoch_dist_means.append(curr_mean)

                                a = each_vectors_dist_from_group_avg
                                a -= tf.reduce_min(a)
                                a = tf.divide(a, tf.math.maximum(1, tf.reduce_max(a))).numpy().tolist()
                                weight_values += a

                                each_vectors_dist_from_group_avg = tf.reduce_mean(each_vectors_dist_from_group_avg)
                                epoch_dist_means.append(each_vectors_dist_from_group_avg)

                            with tf.summary.create_file_writer(logdir + "/train").as_default(epoch):
                                tf.summary.scalar("mean class dist to centers", data=np.mean(epoch_dist_means), step=epoch)

                            print(f"    train mean class dist to centers: {np.mean(epoch_dist_means)}")
                            print(f"    np.mean(weight_values): {np.mean(weight_values)}")

                            weight_values = np.asarray(weight_values, dtype=np_dtype) * 5
                            with tf.summary.create_file_writer(logdir + "/train").as_default(epoch):
                                tf.summary.scalar("mean weight values", data=np.mean(weight_values), step=epoch)

                                # del preds, each_vectors_dist_from_group_avg, curr_preds, mega_loc, mean_loc, a
                            return weight_values

                    # from itertools import repeat
                    # args = zip(repeat(model), repeat(txL), repeat(batch_size), repeat(y_dict_count), repeat(logdir), repeat(epoch), repeat(np_dtype), [0])
                    # weight_values = mp.Pool(1).starmap(train3_proc.run_the_train_metric, args)
                    weight_values = run_the_train_metric()
                    # timerD("Ran the train metric!")
                    # weight_values = quick_save(weight_values, 'weight_values')
                    # exit()

                # weight_values = quick_load('weight_values')

                # tf.keras.backend.clear_session()
                # _ = gc.collect()
                curr = model.fit(x=txL, y=tyL,
                                 sample_weight=weight_values,
                                 epochs=inner_epochs,  # steps_per_epoch=5,
                                 batch_size=train_batch_size,
                                 verbose=1,
                                 validation_data=(t_testxL, t_testyL),
                                 # callbacks=[tensorboard_callback]
                                 )
                results = curr.history

                # print(type(results))
                for k, v in results.items():
                    v = v[0:1]
                    for counter, _v in enumerate(v):
                        with tf.summary.create_file_writer(logdir + "/metrics").as_default(epoch):
                            tf.summary.scalar(k, data=_v, step=epoch)

                    print(k, v)

                time_taken = timerD(f"Finished epoch {epoch}!")
                print("\n\n")

                # model.save_weights("model_new_save.h5")

        print("\n" * 10)
        _ = gc.collect()
        # exit()

    model.save_weights(save_weight_path_all_done)
    timerD(f"  Training is done!")


draw = False
draw = True
if draw:
    with open(wavelet_testfile, 'rb') as datafile:
        testdata = pickle.load(datafile)

    selections = ["quadricycle", "boat", "automobile", "car", "boy", "girl"]
    to_show = 1
    if False:
        pass
    ############################################################################################################
    ############################################################################################################
    ############################################################################################################
    ############################################################################################################

    selections = []
    # selections += ["boy", "girl"]
    selections += ["man", "woman"]
    selections += ["himself"]
    selections += ["statesman"]
    selections += ["youre"]
    selections += ["louise"]
    selections += ["family"]
    selections += ["cofounder"]
    selections += ["inventor"]
    selections += ["kristina"]
    selections += ["romantic"]
    # selections += ["quadricycle"]
    selections += ["discovered"]
    selections += ["arrest"]
    selections += ["showgirls"]
    selections += ["guitarist"]
    # selections += ["king", "queen"]
    # selections += ["bear", "tension"]
    # selections += ["boys"]
    # selections += ["girls"]
    # selections += ["karadjordjevic"]

    # for _ in range(4):
    #     add_this = words[np.random.randint(0, len(words))]
    #     selections += [add_this]

    # selections += ["man", "woman"]
    # selections += ["car", "machine"]

    ############################################################################################################
    ############################################################################################################
    ############################################################################################################
    the_data = [testdata[selection] for selection in selections]

    cluster_averages = np.zeros(shape=len(testdata), dtype=object)
    cluster_stdevs = np.zeros(shape=len(testdata), dtype=object)

    # mank, manv = the_data["man"]
    #
    # x = tf.concat([*v], axis=0)
    # x = tf.reshape(x, shape=(int(x.shape[0] / 2), 2, 105, 28))
    # pred = model(x)  # Should perform it all vectorized

    locs_names = []
    for idx, (k, v) in enumerate(zip(selections, the_data)):
        x = tf.concat([*v], axis=0)
        x = tf.reshape(x, shape=(int(x.shape[0] / 2), 2, 105, 28))
        pred = model(x)  # Should perform it all vectorized

        # mean = tf.reduce_mean(pred, axis=0)
        # mean_loc = mean
        # cluster_averages[idx] = mean
        #
        # mean = tf.expand_dims(mean, axis=0)
        # mean = tf.repeat(mean, pred.shape[0], axis=0)
        # pred_dists = tf.subtract(pred, mean)
        #
        # dists = []
        # for each, e0 in zip(pred_dists, pred):
        #     each = tf.abs(each)
        #     each = tf.reduce_mean(each)
        #     dists.append(each)
        # dists.sort()
        # thresh2 = dists[:to_show][-1]
        #
        # new_pred = []
        # for each, e0 in zip(pred_dists, pred):
        #     each = tf.abs(each)
        #     each = tf.reduce_mean(each)
        #     if each <= thresh2:
        #         if True:
        #             print(f"  k: {k}, each: {each}, thresh2: {thresh2}")
        #         new_pred.append(e0)
        # # pred = new_pred

        # new_pred = tf.reduce_mean(new_pred)
        # locs_names.append([k, new_pred])
        mean_loc = tf.reduce_mean(pred, axis=0)
        locs_names.append([k, mean_loc])
        # return

        mean = tf.expand_dims(mean_loc, axis=0)
        mean = tf.repeat(mean, pred.shape[0], axis=0)
        pred_dists = tf.subtract(pred, mean)
        std = tf.math.reduce_std(pred_dists)
        cluster_stdevs[idx] = std

        print(f"\n  {k}, std: {std}, \n real mean loc: {tf.reduce_mean(mean_loc)}")
        # print(f"\n  {k}, std: {std} \n {mean_loc} \n real mean: {tf.reduce_mean(mean_loc)}")
        # print(f"\n  k: {k}, mean_loc: \n{tf.round(mean_loc * 10000, 4) / 10000}")

    locs_names = np.array(locs_names, dtype=object)
    locs, names = locs_names[:, 1:], locs_names[:, 0]

    new_loc_names = []
    for idx, (loc, name) in enumerate(zip(locs, names)):
        locs[idx] = list(loc)
        for idx1, each in enumerate(locs[idx]):
            # if idx1 % 2 == 0:
            new_loc_names.append([name, each])
        # print(f"  loc: {len(loc)}")

    locs_names = np.array(new_loc_names, dtype=object)
    locs, names = locs_names[:, 1], locs_names[:, 0]

    locs = list(locs)
    names = list(names)
    locs = np.squeeze(locs)
    names = np.squeeze(names)
    print()
    print()
    print()
    # print(f"  len(locs): {locs.shape}")
    # print(f"  len(names): {names.shape}")

    # timerD("  Starting TSNE...")

    print(f"       locs: {locs.shape}")
    # for each in locs:
    #     print(each)
    print("\n" * 3)
    # return

    print(f"          TSNE:")

    counter = 0
    save_dir = f"images/{len(selections)}-{counter}"
    while os.path.exists(save_dir):
        counter += 1
        save_dir = f"images/{len(selections)}-{counter}"
    save_dir = f"images/{len(selections)}"

    os.system(f"mkdir -p {save_dir}")

    # for perplexity in range(10, 5000, 30):
    scaler = 1
    scale = 10 ** (-scaler)
    # perplexities = np.arange(0, 1, scale)
    # perplexities = np.arange(0.2, 0.20001, scale)
    # perplexities = np.arange(1, 5, scale)
    perplexities = np.arange(1, 2, scale)

    LR_scaler = -1
    LR_scale = 10 ** (-LR_scaler)
    learning_rates = np.arange(10, 11, LR_scale)
    # learning_rates = np.arange(0, 100, LR_scale)

    # print(f"  perplexities: {perplexities}")
    # print(f"  learning_rates: {learning_rates}")
    # # for perplexity in range(1, 50, 1):
    # counter_name = 0
    # for perplexity in np.arange(1, 2, 0.25):
    #     perplexity = np.round(perplexity, 2)
    #     # perplexity = np.round(perplexity, scaler)
    #     for learning_rate in learning_rates:
    #         learning_rate = np.round(learning_rate, LR_scaler)
    #
    #         print(f"                 perplexity: {perplexity}, learning_rate: {learning_rate}", end='', flush=True)
    #         # tsne = TSNE(perplexity=perplexity, n_jobs=mp.cpu_count(), random_state=42, angle=0.9).fit_transform(locs)
    #         # tsne = TSNE(perplexity=perplexity, n_jobs=mp.cpu_count(), random_state=42, learning_rate=learning_rate).fit_transform(locs)
    #         tsne = TSNE(perplexity=perplexity, n_jobs=mp.cpu_count(), learning_rate=learning_rate).fit_transform(locs)
    #         tsne = np.array(tsne, dtype=object)
    #
    #         for label, x, y in zip(names, tsne[:, 0], tsne[:, 1]):
    #             plt.annotate(label, xy=(x, y), xytext=(-8, 3), textcoords="offset points", annotation_clip=False)
    #         plt.scatter(tsne[:, 0], tsne[:, 1])
    #
    #         save_path = f"{save_dir}/Num-{counter_name}__P-{perplexity}_LR-{learning_rate}.png"
    #         while os.path.exists(save_path):
    #             counter_name += 1
    #             save_path = f"{save_dir}/Num-{counter_name}__P-{perplexity}_LR-{learning_rate}.png"
    #
    #         plt.title(f"Perplexity: {perplexity}, Learning Rate: {learning_rate}")
    #         plt.savefig(save_path)
    #         # plt.show()
    #         plt.clf()
    #         print(f", saved to {save_path}!")

    # print()
    for w, l in zip(selections, tsne):
        l = f"{l}"[1:-1]
        l = l.split()
        new_l = []
        for each in l:
            new_l.append(float(each))
        l = new_l

        # l = float(l)
        # l = np.round(l, 2)
        # l = str(l)[1:-1]
        # l = l.split()
        # new_l = []
        # for each in l:
        #     new_l.append(f"{str(each):<5}")
        # l = new_l
        # l = str(l)
        # l = l.replace('\'', '')
        print(f"  {w:>20} [ ", end='')
        idx = 0
        for each in l:
            if idx == 0:
                z = f"{each:3.2f}"
                print(f"{z:>8}", end=', ')
            else:
                z = f"{each:3.2f}"
                print(f"{z:>8} ]")
            idx += 1
        # print(f"  {w:<20} {l}")

make_dict_for_Nate = False
# make_dict_for_Nate = True
if make_dict_for_Nate:  # if creating the output dictionary

    os.system("clear")
    with open(wavelet_trainfile, 'rb') as datafile:
        wavelet_trainfile_data = pickle.load(datafile)

    data_to_use = wavelet_trainfile_data

    output_dict = {}
    for idx, (k, v) in enumerate(data_to_use.items()):
        print(f"  {k:>40} ({len(v):6d})  ({idx:5d}/{len(data_to_use):5d})")
        output_dict[k] = []
        x = tf.concat([*v], axis=0)
        x = tf.reshape(x, shape=(int(x.shape[0] / 2), 2, 105, 28))
        pred = model(x)  # Should perform it all vectorized
        pred = pred.numpy()
        for each in pred:
            output_dict[k].append(each)
        # if idx > 3:
        #     break

    for idx, (k, v) in enumerate(output_dict.items()):
        print(f"    k: {k}, v: {len(v)}")

    pickle.dump(output_dict, open("predictions_on_wavelets.dict", 'wb'))

load_wavelet_preds = False
# load_wavelet_preds = True
if load_wavelet_preds:
    with open("predictions_on_wavelets.dict", 'rb') as datafile:
        loaded_preds = pickle.load(datafile)

    for idx, (k, v) in enumerate(loaded_preds.items()):
        print(f"  k: {k}, len(v): {len(v)}, {v}")

print("\n" * 10)
# exit()
pass

##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
##############################################################################################
if __name__ == "__main__":
    os.system("clear")

    print_all_words_at_start = False
    # print_all_words_at_start = True

    if print_all_words_at_start:
        flips = False
        flips = True
        for index, word in enumerate(words[:-1]):
            if flips:
                if index != len(words) - 1:
                    print(f" {word}", end=',')
                else:
                    print(word, end='')
                if index % 7 == 0:
                    print()
            else:
                print(word)

    print()
    print()

    for _ in range(1):
        new_plan()
