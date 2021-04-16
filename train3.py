# with open("words_in_vocab.txt", 'w') as f:
#     collection = ' '.join(words)
#     f.write(collection)
# for word in words:
#     f.write()
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
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

# torch.manual_seed(42)
import multiprocessing as mp

import os, pickle

from shared_functions import timerD

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow_addons as tfa

if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')

    try:
        if gpus:
            for gpu in gpus:
                # pass
                tf.config.experimental.set_memory_growth(gpu, True)
    except:
        pass

    tf.config.experimental.enable_tensor_float_32_execution(True)

dim = 50

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


def load_model():
    # inputs = tf.keras.layers.Input(shape=(channel_indxs, 28))
    inputs = tf.keras.layers.Input(shape=(51, 105))
    inputs = tf.keras.layers.Input(shape=(2, 105, 28))
    # inputs = tf.keras.layers.Input(shape=(105, 28))
    nh, kd = 3, 3
    A = tf.keras.layers.MultiHeadAttention(num_heads=nh, key_dim=kd, dropout=0.1)
    B = tf.keras.layers.MultiHeadAttention(num_heads=nh, key_dim=kd, dropout=0.1)
    C = tf.keras.layers.MultiHeadAttention(num_heads=nh, key_dim=kd, dropout=0.1)
    D = tf.keras.layers.MultiHeadAttention(num_heads=nh, key_dim=kd, dropout=0.1)

    # # Path 0
    # x0 = tf.keras.layers.Conv2D(filters=16, kernel_size=3, strides=3, padding='same', )(inputs)
    # # x0 = tf.keras.layers.Conv2D(filters=256, kernel_size=3, strides=3, padding='same', )(inputs)
    # x0 = tf.keras.layers.Activation(tfa.activations.lisht)(x0)
    # x0 = tf.keras.layers.LayerNormalization()(x0)
    # x0 = A(x0, x0, return_attention_scores=False)
    # x0 = A(x0, x0, return_attention_scores=False)
    #
    # # Path 1
    # x1 = tf.keras.layers.Conv2D(filters=32, kernel_size=6, strides=6, padding='same', )(inputs)
    # # x1 = tf.keras.layers.Conv2D(filters=512, kernel_size=6, strides=6, padding='same', )(inputs)
    # x1 = tf.keras.layers.Activation(tfa.activations.lisht)(x1)
    # x1 = tf.keras.layers.LayerNormalization()(x1)
    # x1 = B(x1, x1, return_attention_scores=False)
    # x1 = B(x1, x1, return_attention_scores=False)
    #
    # # Merge
    # x = C(x0, x1, return_attention_scores=False)
    # x = C(x, x1, return_attention_scores=False)
    # Processing
    x = inputs
    x = D(x, x, return_attention_scores=False)

    x = tf.keras.layers.Conv1D(filters=64, kernel_size=2, strides=2, padding='same', )(x)
    x = tf.keras.layers.Activation(tfa.activations.lisht)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv1D(filters=48, kernel_size=2, strides=2, padding='same', )(x)
    x = tf.keras.layers.Activation(tfa.activations.lisht)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv1D(filters=32, kernel_size=2, strides=2, padding='same', )(x)
    x = tf.keras.layers.Activation(tfa.activations.lisht)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv1D(filters=24, kernel_size=2, strides=2, padding='same', )(x)
    x = tf.keras.layers.Activation(tfa.activations.lisht)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Conv1D(filters=16, kernel_size=2, strides=2, padding='same', )(x)
    x = tf.keras.layers.Activation(tfa.activations.lisht)(x)
    x = tf.keras.layers.BatchNormalization()(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(120)(x)
    x = tf.keras.layers.Dense(100)(x)
    # x = tf.keras.layers.Dense(1, activation=None)(x)  # No activation on final dense layer
    x = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)  # L2 normalize embeddings
    outputs = x

    model = tf.keras.models.Model(inputs, outputs)
    # tf.keras.layers.Dense(100, activation=None),  # No activation on final dense layer

    LR = tfa.optimizers.CyclicalLearningRate(1e-4, 1e-2, step_size=2000, scale_fn=lambda x: 0.96)
    optimizer = tfa.optimizers.LAMB(learning_rate=LR)
    # LR = tfa.optimizers.CyclicalLearningRate(1e-3, 1e-1, step_size=2000, scale_fn=lambda x: 0.96)
    model.compile(
        optimizer=optimizer,
        # loss=tfa.losses.TripletSemiHardLoss())
        loss=tfa.losses.TripletHardLoss())
    try:
        model.load_weights(save_weight_path)
        os.system("clear")
        # model.load_weights(save_weight_path, by_name=True, skip_mismatch=True)
    except:
        try:
            model.load_weights(save_weight_path)
        except:
            print("  Oddities whileloading the save.")
    return model


if __name__ == "__main__":
    model = load_model()


class callback3(tf.keras.callbacks.Callback):
    def __init__(self, testfile="data/WD/test_dict.dict"):
        super(callback3, self).__init__()
        self.best_loss = 99999

    def on_epoch_end(self, epoch, logs=None):
        global testdata
        timerD("", silent=True)
        # timerD(f"\r  Finished epoch {epoch}.")
        print(f"  on_epoch_end() callback:")
        if logs is not None:
            for k, v in logs.items():
                print(f"  {k}: {v}")

        self.model.save_weights(save_weight_path_last_epoch)

        if 'val_loss' in logs:
            loss_to_save_on = logs['val_loss']
        else:
            loss_to_save_on = logs['loss']

        if loss_to_save_on <= self.best_loss:
            self.best_loss = loss_to_save_on
            self.model.save_weights(save_weight_path)
            print(f"    New best loss found! {loss_to_save_on}")

            timerD("Finished on_epoch_end.")
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
    if not os.path.exists(train_dataset_path):
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

model = load_model()
def new_plan():
    # os.system("clear")

    timerD(f"  Finished with imports")

    ## Prepare the train dataset
    create_train_dataset()
    with open(train_dataset_elementspec_path, 'rb') as datafile:
        trainspec = pickle.load(datafile)
    train_dataset = tf.data.experimental.load(train_dataset_path, trainspec)
    train_dataset = train_dataset.batch(7500)
    timerD("Loaded the train dataset from the disk!")

    ## Prepare the test dataset
    create_test_dataset()
    with open(test_dataset_elementspec_path, 'rb') as datafile:
        testspec = pickle.load(datafile)
    test_dataset = tf.data.experimental.load(test_dataset_path, testspec)
    test_dataset = test_dataset.batch(7500)
    timerD("Loaded the test dataset from the disk!")

    # spec = "(TensorSpec(shape=(2, 105, 28), dtype=tf.float32, name=None), TensorSpec(shape=(), dtype=tf.int32, name=None))"
    # model = load_model()
    summarize_model = False
    if summarize_model:
        # model.build(input_shape=(None, *np.array(x[0]).shape))
        history = model.fit(
            train_dataset,
            verbose=0,
            steps_per_epoch=1,
            epochs=1)
        # history = model.fit(
        #     train_dataset, steps_per_epoch=1,
        #     epochs=5)
        model.summary()
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
    if do_training:
        os.system("mkdir -p WD/tb")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="WD/tb", histogram_freq=1)

        # try:
        #     model.load_weights(save_weight_path)
        # except:
        #     print(f"  failed to load weights at {save_weight_path}")

        history = model.fit(
            # x, y,
            train_dataset,
            validation_data=test_dataset,
            # batch_size=4096,
            # batch_size=4194304,
            batch_size=140,
            epochs=1000,
            callbacks=[callback3(),
                       # tensorboard_callback,
                       ],
        )

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
        # selections += ["king", "queen"]
        # selections += ["bear", "tension"]
        # selections += ["boys"]
        # selections += ["girls"]
        # selections += ["karadjordjevic"]

        for _ in range(8):
            add_this = words[np.random.randint(0, len(words))]
            selections += [add_this]

        # selections += ["man", "woman"]
        # selections += ["car", "machine"]

        ############################################################################################################
        ############################################################################################################
        ############################################################################################################
        the_data = [testdata[selection] for selection in selections]

        cluster_averages = np.zeros(shape=len(testdata), dtype=object)
        cluster_stdevs = np.zeros(shape=len(testdata), dtype=object)

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
        scaler = 0
        scale = 10 ** (-scaler)
        # perplexities = np.arange(0, 1, scale)
        # perplexities = np.arange(0.2, 0.20001, scale)
        # perplexities = np.arange(1, 5, scale)
        perplexities = np.arange(2, 2.0001, scale)

        LR_scaler = -1
        LR_scale = 10 ** (-LR_scaler)
        learning_rates = np.arange(10, 11, LR_scale)
        # learning_rates = np.arange(0, 100, LR_scale)

        print(f"  perplexities: {perplexities}")
        print(f"  learning_rates: {learning_rates}")
        # for perplexity in range(1, 50, 1):
        counter_name = 0
        for perplexity in perplexities:
            for learning_rate in learning_rates:
                perplexity = np.round(perplexity, scaler)
                learning_rate = np.round(learning_rate, LR_scaler)

                print(f"                 perplexity: {perplexity}, learning_rate: {learning_rate}", end='', flush=True)
                # tsne = TSNE(perplexity=perplexity, n_jobs=mp.cpu_count(), random_state=42, angle=0.9).fit_transform(locs)
                # tsne = TSNE(perplexity=perplexity, n_jobs=mp.cpu_count(), random_state=42, learning_rate=learning_rate).fit_transform(locs)
                tsne = TSNE(perplexity=perplexity, n_jobs=mp.cpu_count(), learning_rate=learning_rate).fit_transform(locs)
                tsne = np.array(tsne, dtype=object)

                for label, x, y in zip(names, tsne[:, 0], tsne[:, 1]):
                    plt.annotate(label, xy=(x, y), xytext=(-3, 1), textcoords="offset points", annotation_clip=False)
                plt.scatter(tsne[:, 0], tsne[:, 1])

                save_path = f"{save_dir}/P-{perplexity}_LR-{learning_rate} - {counter_name}.png"
                while os.path.exists(save_path):
                    counter_name += 1
                    save_path = f"{save_dir}/P-{perplexity}_LR-{learning_rate} - {counter_name}.png"

                plt.title(f"Perplexity: {perplexity}, Learning Rate: {learning_rate}")
                plt.savefig(save_path)
                # plt.show()
                plt.clf()
                print(f", saved to {save_path}!")

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

    make_dict_nate = False
    # make_dict_nate = True
    if make_dict_nate:  # if creating the output dictionary

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
    print_all_words_at_start = True

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

    for _ in range(1000):
        new_plan()
