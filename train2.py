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

# torch.manual_seed(42)
import multiprocessing as mp

import os

from shared_functions import timerD

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow_addons as tfa

if __name__ == '__main__':
    gpus = tf.config.list_physical_devices('GPU')

    try:
        if gpus:
            for gpu in gpus:
                pass
                # tf.config.experimental.set_memory_growth(gpu, True)
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

# wavelet_trainfile += ".npy"
# wavelet_testfile += ".npy"
# wavelet_devfile += ".npy"

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


class eeg2vec(tf.keras.models.Model):
    # Class that defines our model
    def __init__(self, batch_size=None, channel_indxs=None, time_window=None):
        super(eeg2vec, self).__init__()
        self.channel_indxs = channel_indxs
        self.batch_size = batch_size
        # self.hidden_dim = hidden_dim

        # input = tf.keras.layers.Input()
        # self.cAconv1 = nn.Conv1d(time_window, 38, 5)
        # self.cDconv1 = nn.Conv1d(time_window, 38, 5)
        #
        # self.cAmaxpool1 = nn.MaxPool1d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        # self.cDmaxpool1 = nn.MaxPool1d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        #
        # self.cAconv2 = nn.Conv1d(len(channel_indxs), hidden_dim)
        # self.cDconv2 = nn.Conv1d(channels, hidden_dim)
        #
        # self.cAconv1 = nn.Conv1d(len(channel_indxs), hidden_dim)
        # self.cDconv1 = nn.Conv1d(channels, hidden_dim)

    '''     
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
    '''

    # This is the forward computation, which constructs the computation graph
    # def forward(self, rawEEG):
    #     cAs = []
    #     cDs = []
    #     for N in range(len(rawEEG)):
    #         # channel_data = np.transpose(rawEEG)
    #         channel_data = rawEEG
    #         cA, cD = pywt.dwt(channel_data, 'coif1')
    #         cAs.append(cA)
    #         cDs.append(cD)
    #     cAouts1 = self.cAconv1(torch.tensor(cAs, dtype=torch.float32))
    #     cDouts1 = self.cAconv1(torch.tensor(cDs, dtype=torch.float32))
    #     '''
    #     # Get the embeddings
    #     embeds = self.word_embeddings(sentence)
    #     # put them through the LSTM and get its output
    #     lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
    #     # pass those outputs through the linear layer
    #     tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
    #     # convert the logits to a log probability distribution
    #     tag_scores = F.log_softmax(tag_space, dim=1)
    #     return tag_scores
    #     '''


# Acquired from https://www.kaggle.com/hirotaka0122/triplet-loss-with-pytorch
def calc_euclidean(x1, x2):
    return (x1 - x2).pow(2).sum(1)


# class TripletLoss(nn.Module):
#     def __init__(self, margin=1.0):
#         super(TripletLoss, self).__init__()
#         self.margin = margin
#
#     def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
#         distance_positive = self.calc_euclidean(anchor, positive)
#         distance_negative = self.calc_euclidean(anchor, negative)
#         losses = torch.relu(distance_positive - distance_negative + self.margin)
#
#         return losses.mean()


# @tf.function(experimental_compile=True)
# def TripletLossFunc(anchor, positive_and_negative, margin=1.0):
#     positive, negative = positive_and_negative[:15], positive_and_negative[15:]
#     distance_positive = tf.norm(anchor - positive)
#     distance_negative = tf.norm(anchor - negative)
#     losses = distance_positive - distance_negative + margin
#     if losses < 0:
#         losses = 0.0
#     avg_losses = tf.reduce_mean(losses)
#     return avg_losses


def preprocess(batch):
    cAs = []
    cDs = []
    batch = np.transpose(batch)
    for N in batch:
        # channel_data = np.transpose(N)
        # cA, cD = pywt.dwt(channel_data, 'coif1')
        cA, cD = pywt.dwt(N, 'coif1')
        cAs.append(cA)
        cDs.append(cD)
    cAs = np.array(cAs, dtype=np.float32)
    cDs = np.array(cDs, dtype=np.float32)
    return cAs, cDs
    # return torch.tensor(cAs, dtype=torch.float32), torch.tensor(cDs, dtype=torch.float32)


def get_next_element(anchors):
    # shuffled_anchors = anchors.copy()
    # np.random.shuffle(shuffled_anchors)
    shuffled_anchors = anchors

    for anc_idx, anchor in enumerate(shuffled_anchors):
        anchorEEG = data[anchor[0]][anchor[1]]

        # the options for positive are any data point with a matching word to the anchor
        posOptions = [pos for pos in shuffled_anchors if pos[0] == anchor[0]]
        # the positive is chosen as the data point that falls circularly after the anchor
        pos = posOptions[(posOptions.index(anchor) + 1) % len(posOptions)]
        posEEG = data[pos[0]][pos[1]]

        # the negative is the next data point (circularly) which has a different word than the anchor
        neg_idx = shuffled_anchors.index(anchor) + 1
        neg = shuffled_anchors[neg_idx % len(shuffled_anchors)]
        while neg[0] == anchor[0]:
            neg_idx += 1
            neg = shuffled_anchors[neg_idx % len(shuffled_anchors)]
        negEEG = data[neg[0]][neg[1]]

        # add the data points to the batch

        # all_non_empty_bool = np.product(list(map(lambda x: len(x) > 0, [anchorEEG, posEEG, negEEG])))

        a, aa = preprocess(anchorEEG)
        p, pp = preprocess(posEEG)
        n, nn = preprocess(negEEG)

        x = tf.stack([*[a, aa], *[p, pp], *[n, nn]], axis=0)
        x = tf.expand_dims(x, axis=0)

        yield x


# Tensorflow-style generator that grabs a random element
def train_data_generator(unused_dummy_argument=None):
    global anchors

    while True:
        # if True:
        anchor_idx = np.random.randint(0, len(anchors))
        anchor = anchors[anchor_idx]

        word, content_idx, data_indice = anchor[0], anchor[1], anchor[2]
        # anchorEEG = data[word][content_idx]

        # the options for positive are any data point with a matching word to the anchor
        # the positive is chosen as the data point that falls circularly after the anchor
        pos_indice = (content_idx + 1) % len(data[word])
        # posEEG = data[word][pos_indice]

        # the negative is the next data point (circularly) which has a different word than the anchor
        neg_data_idx = (data_indice + 1) % len(list(data.keys()))
        neg_word = list(data.keys())[neg_data_idx]
        neg_contents_idx = np.random.randint(0, len(data[neg_word]))
        # negEEG = data[neg_word][neg_contents_idx]

        a, aa = wavelet_trainfile_data[word][content_idx]
        p, pp = wavelet_trainfile_data[word][pos_indice]
        n, nn = wavelet_trainfile_data[neg_word][neg_contents_idx]

        # a, aa = preprocess(anchorEEG)
        # p, pp = preprocess(posEEG)
        # n, nn = preprocess(negEEG)

        x = tf.stack([*[a, aa], *[p, pp], *[n, nn]], axis=0)

        # print(f"  x.shape: {x.shape}")
        # print(f"  a.shape: {a.shape}")
        # print(f"  anchorEEG.shape: {anchorEEG.shape}")
        # x = tf.expand_dims(x, axis=0)

        # print(f"  generator() called!")
        # print(f"    word: ({word}, {content_idx}), pos: ({word}, {pos_indice}), neg: ({neg_word}, {neg_contents_idx})")
        # print(f"    return shape: {x.shape}")
        # print(f"    tf.reduce_mean(x): {tf.reduce_mean(x)}")

        # return x
        yield x


# Tensorflow-style generator that grabs a random element
def validation_data_generator(unused_dummy_argument=None):
    global anchors

    anchor_idx = np.random.randint(0, 100)
    anchor = anchors[anchor_idx]

    word, content_idx, data_indice = anchor[0], anchor[1], anchor[2]
    # anchorEEG = data[word][content_idx]

    pos_indice = (content_idx + 1) % len(data[word])
    # posEEG = data[word][pos_indice]

    neg_data_idx = (data_indice + 1) % len(list(data.keys()))
    neg_word = list(data.keys())[neg_data_idx]
    neg_contents_idx = np.random.randint(0, len(data[neg_word]))
    # negEEG = data[neg_word][neg_contents_idx]

    # a, aa = preprocess(anchorEEG)
    # p, pp = preprocess(posEEG)
    # n, nn = preprocess(negEEG)

    a, aa = wavelet_testfile_data[word][content_idx]
    p, pp = wavelet_testfile_data[word][pos_indice]
    n, nn = wavelet_testfile_data[neg_word][neg_contents_idx]

    x = tf.stack([*[a, aa], *[p, pp], *[n, nn]], axis=0)
    # x = tf.expand_dims(x, axis=0)

    # print(f"  generator() called!")
    # print(f"    word: ({word}, {content_idx}), pos: ({word}, {pos_indice}), neg: ({neg_word}, {neg_contents_idx})")
    # print(f"    return shape: {x.shape}")
    # print(f"    tf.reduce_mean(x): {tf.reduce_mean(x)}")

    yield x


def train_gen(input=None):
    return tf.data.Dataset.from_generator(train_data_generator, output_types=tf.float32)


def validation_gen(input=None):
    return tf.data.Dataset.from_generator(validation_data_generator, output_types=tf.float32)


def new_plan():
    os.system("clear")
    timerD(f"  new_plan()")
    # flipped_data = np.zeros(shape=(len(data.values())),dtype=object)

    flipped_data = []

    # wavelet_trainfile_data
    data_to_use = data
    data_to_use = wavelet_trainfile_data

    # Vectorize the keys
    key_to_vector_dict = {}
    vector_to_key_dict = {}
    for idx0, (k, v) in enumerate(data_to_use.items()):
        if k not in key_to_vector_dict:
            val = len(key_to_vector_dict)
            key_to_vector_dict[k] = len(key_to_vector_dict)
            vector_to_key_dict[val] = k

    for idx0, (k, v) in enumerate(data_to_use.items()):
        for idx1, each in enumerate(v):
            x = [each, key_to_vector_dict[k]]
            flipped_data.append(x)

    timerD(f"  data preprocessing complete!")
    flipped_data = flipped_data[:1000]

    flipped_data = np.array(flipped_data, dtype=object)
    x, y = flipped_data[:, 0], flipped_data[:, 1]
    # x = np.array(x.tolist(),dtype=np.float32)
    # y = np.array(y.tolist(),dtype=np.float32)
    x, y = x.tolist(), y.tolist()
    # x = tf.convert_to_tensor(x,dtype=tf.float32)
    # y = tf.convert_to_tensor(y,dtype=tf.float32)

    # x = np.transpose(x).tolist()
    train_dataset = tf.data.Dataset.from_tensor_slices((x, y))
    # timerD(f"  tensors created!")
    train_dataset = train_dataset.shuffle(1024).batch(32)
    timerD(f"  train_dataset is ready!")

    # inputs = tf.keras.layers.Input(shape=(channel_indxs, 28))
    inputs = tf.keras.layers.Input(shape=(51, 105))
    inputs = tf.keras.layers.Input(shape=(2, 105, 28))
    nh, kd = 3, 3
    A = tf.keras.layers.MultiHeadAttention(num_heads=nh, key_dim=kd, dropout=0.1)
    B = tf.keras.layers.MultiHeadAttention(num_heads=nh, key_dim=kd, dropout=0.1)
    C = tf.keras.layers.MultiHeadAttention(num_heads=nh, key_dim=kd, dropout=0.1)
    D = tf.keras.layers.MultiHeadAttention(num_heads=nh, key_dim=kd, dropout=0.1)

    # Path 0
    x0 = tf.keras.layers.Conv2D(filters=32, kernel_size=3, strides=3, padding='same', )(inputs)
    x0 = tf.keras.layers.Activation(tfa.activations.lisht)(x0)
    x0 = tf.keras.layers.LayerNormalization()(x0)
    x0 = A(x0, x0, return_attention_scores=False)
    x0 = A(x0, x0, return_attention_scores=False)

    # Path 1
    x1 = tf.keras.layers.Conv2D(filters=64, kernel_size=6, strides=6, padding='same', )(inputs)
    x1 = tf.keras.layers.Activation(tfa.activations.lisht)(x1)
    x1 = tf.keras.layers.LayerNormalization()(x1)
    x1 = B(x1, x1, return_attention_scores=False)
    x1 = B(x1, x1, return_attention_scores=False)

    # Merge
    x = C(x0, x1, return_attention_scores=False)
    x = C(x, x1, return_attention_scores=False)

    # Processing
    x = D(x, x, return_attention_scores=False)

    x = tf.keras.layers.Conv1D(filters=8, kernel_size=2, strides=2, padding='same', )(x)
    x = tf.keras.layers.Activation(tfa.activations.lisht)(x)
    x = tf.keras.layers.LayerNormalization()(x)

    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(120)(x)
    x = tf.keras.layers.Dense(100)(x)
    # x = tf.keras.layers.Dense(1, activation=None)(x)  # No activation on final dense layer
    x = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)  # L2 normalize embeddings
    outputs = x

    model = tf.keras.models.Model(inputs, outputs)
    # tf.keras.layers.Dense(100, activation=None),  # No activation on final dense layer

    # model.build(input_shape=(None, *np.array(x[0]).shape))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss=tfa.losses.TripletSemiHardLoss())
    history = model.fit(
        # x, y,
        train_dataset,
        steps_per_epoch=1,
        epochs=5)
    # history = model.fit(
    #     train_dataset, steps_per_epoch=1,
    #     epochs=5)
    model.summary()

    timerD(f"  Ready for the main fit...")
    # import tensorflow_datasets as tfds
    # train_dataset, test_dataset = tfds.load(name="mnist", split=['train', 'test'], as_supervised=True)

    # print(f"  x[0].shape: {np.array(x[0]).shape}")
    # print(f"  y[0].shape: {np.array(y[0]).shape}")
    # print(f"  x[0]: {x[0]}")
    # print(f"  y[0]: {y[0]}")
    # print("\n" * 4)
    history = model.fit(
        # x, y,
        train_dataset,
        epochs=5)
    timerD(f"  len(flipped_data) == {len(flipped_data)}")
    exit()
    pass


def main():
    global anchors, data, test_anchors, testdata, wavelet_trainfile_data, wavelet_testfile_data
    import TFModel
    # os.system("clear")
    # filename = "D:\\NLP Datasets\\word_dict_new.dict"
    # filename = "data/WD/SR_NR2_TSR2_lower_numless_word_dict.dict"

    #########################
    ### Load the trainfile ##
    # Sorts the dictionary, but in one line to be 100% safe from memory usage issues
    # data = np.load(trainfile, allow_pickle=True, fix_imports=False)
    data = np.load(trainfile, allow_pickle=True, fix_imports=False)
    train = data
    data = {e[0]: e[1] for e in sorted([[k, v] for k, v in data.items()])}

    anchors = []

    for idx, word in enumerate(data):
        for eeg_idx in range(len(train[word])):
            anchors.append((word, eeg_idx, idx))

    import pickle
    with open(wavelet_trainfile, 'rb') as datafile:
        wavelet_trainfile_data = pickle.load(datafile)

    import pickle
    with open(wavelet_testfile, 'rb') as datafile:
        wavelet_testfile_data = pickle.load(datafile)
    #########################

    # ########################
    # ########################
    # ### Load the testfile ##
    # with open(testfile, 'rb') as datafile:
    #     testdata = pickle.load(datafile)
    #
    # testdata = {e[0]: e[1] for e in sorted([[k, v] for k, v in testdata.items()])}
    #
    # test_anchors = []
    # for idx, word in enumerate(testdata):
    #     for eeg_idx in range(len(testdata[word])):
    #         test_anchors.append((word, eeg_idx, idx))
    # #########################

    ##############
    ##############
    ## Presets  ##
    epochs = 10000
    batch_size = 32
    loss_margin = 1
    embedding_dim = 100
    # steps_per_epoch = 5000
    # steps_per_epoch = len(data)
    steps_per_epoch = len(anchors)
    # steps_per_epoch = 2500
    validation_steps_per_epoch = int(steps_per_epoch / 2)
    MODEL_FILE = "NLP Models\\clustering_model_v001.pt"
    word_embeddings_file = "NLP Models\\word_embeddings_v001.dict"
    ##############

    # Prepares the data
    x = tf.data.Dataset.range(12)
    x = x.interleave(map_func=train_gen,
                     num_parallel_calls=tf.data.AUTOTUNE,
                     # num_parallel_calls=1,
                     cycle_length=tf.data.AUTOTUNE,
                     block_length=1,
                     deterministic=False)
    x = x.prefetch(tf.data.AUTOTUNE)
    x = x.repeat()  # Random without repeats
    x2 = x.as_numpy_iterator()

    # Creates the model
    model = TFModel.tf_model(channel_indxs=105)
    model.compile(tfa.optimizers.LAMB(), steps_per_execution=200)

    # Automatically compute the shape
    input_shape = (None, *next(x2).shape[1:])
    model.encoder.build(input_shape=input_shape)

    print("\n")
    model.encoder.summary()
    print("\n")

    # import wandb
    # from wandb.keras import WandbCallback
    # wandb.init(config={"hyper": "parameter"})

    try:
        save_weight_path = "D:\\NLP Datasets\\TFModel_encoder.weights"  # Warning: must be changed in TFModel.on_epoch_end
        if not os.path.exists(save_weight_path):
            save_weight_path = "TFModel_encoder.weights"
        model.encoder.load_weights(save_weight_path)
    except Exception as e:
        print("*" * 40, flush=True)
        print("*" * 40, flush=True)
        print(f"  Failed to load the model weights at {save_weight_path}. \n  e: {e}", flush=True)

    # # os.system("clear")
    # y = np.load(new_train_data_path, allow_pickle=True)
    # # print(y)
    # # print("\n\n\n")
    # y = y.tolist()
    # # print(y)
    # # print("\n\n\n")
    # y = tf.data.Dataset.from_tensor_slices(y).repeat()
    # print(y)
    # print("\n\n\n")
    # with open(wavelet_devfile, 'rb') as datafile:
    #     wavelet_devfile_data = pickle.load(datafile)
    # wavelet_devfile_data = {e[0]: e[1] for e in sorted([[k, v] for k, v in wavelet_devfile_data.items()])}

    # import pickle
    # with open(wavelet_devfile, 'rb') as datafile:
    #     wavelet_devfile_data = pickle.load(datafile)
    # TFModel.validation_accuracy(model, wavelet_devfile_data)
    # exit()

    new_plan()
    exit()
    model.fit(
        x,
        # y,
        steps_per_epoch=steps_per_epoch, epochs=epochs,
        # steps_per_epoch=5000, epochs=epochs,
        batch_size=batch_size,
        verbose=1,
        # validation_data=y, validation_steps=len(testdata),
        use_multiprocessing=True, workers=mp.cpu_count(),
        callbacks=[TFModel.validation_callback()],
    )
    # model.encoder.save("TFModel_encoder.save")
    # model.encoder.save_weights("TFModel_encoder.weights")
    return

    # print("Saving model...")
    # torch.save(model, MODEL_FILE)
    #
    # with open(testfile, 'rb') as datafile:
    #     testdata = pickle.load(datafile)
    # # temp = validation_accuracy(model, testdata, embedding_dim)
    # temp = 0
    # print("Test Accuracy:", temp)
    # with torch.no_grad():
    #     word_embeddings = {}
    #     for word in data:
    #         # Find the average of a cluster's embeddings
    #         word_sum = torch.zeros([dim], dtype=torch.float32)
    #         for eeg in data[word]:
    #             cAs, cDs = preprocess([eeg])
    #             word_sum += model(cAs, cDs)
    #         word_avg = torch.div(word_sum, len(data[word]))
    #         word_embeddings[word] = word_avg
    #
    # pickle.dump(word_embeddings, open(word_embeddings_file, 'wb'))


if __name__ == "__main__":
    main()
