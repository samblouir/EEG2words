import importlib

import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pywt as wt
import pickle
import numpy as np

from __main__ import data
# torch.manual_seed(42)
import multiprocessing as mp

import os

from shared_functions import timerD

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import tensorflow_addons as tfa

dim = 50


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


@tf.function(experimental_compile=True)
def TripletLossFunc(anchor, positive_and_negative, margin=1.0):
    positive, negative = positive_and_negative[:15], positive_and_negative[15:]
    distance_positive = tf.norm(anchor - positive)
    distance_negative = tf.norm(anchor - negative)
    losses = distance_positive - distance_negative + margin
    if losses < 0:
        losses = 0.0
    avg_losses = tf.reduce_mean(losses)
    return avg_losses


# # return the score of the model clustering over the data
# def validation_accuracy(model, dev, dim, S=1, C=1, quiet=True):
#     with torch.no_grad():
#         cluster_averages = []
#         cluster_stdevs = []
#         for word in dev:
#
#             # Find the average of a clusters embeddings
#             word_sum = torch.zeros([dim], dtype=torch.float32)
#             for eeg in dev[word]:
#                 cAs, cDs = preprocess([eeg])
#                 word_sum += model(cAs, cDs)
#             word_avg = torch.div(word_sum, len(data[word]))
#
#             # Find the standard deviation from the clusters average
#             dist_sum = 0
#             for eeg in dev[word]:
#                 cAs, cDs = preprocess([eeg])
#                 dist_sum += calc_euclidean(model(cAs, cDs), word_avg)
#             word_stdev = dist_sum / fix_count
#             cluster_averages.append(word_avg)
#             cluster_stdevs.append(word_stdev)
#
#         # Calculate a score for the separation of clusters (Higher is better)
#         dist_count = 0
#         dist_sum = 0
#         for average1 in cluster_averages:
#             for average2 in cluster_averages:
#                 if average1 is not average2:
#                     dist_sum += calc_euclidean(average1, average2)
#                     dist_count += 1
#         separation_score = dist_sum / dist_count
#         if not quiet:
#             print("Separation:", separation_score)
#         # Calculate a score for the compactness of clusters (Lower is better)
#         compactness_score = sum(cluster_stdevs) / len(cluster_stdevs)
#         if not quiet:
#             print("Compactness:", compactness_score)
#         # The total score is separation over compactness weighted by S and C
#         return (S * separation_score) / (C * compactness_score)

def preprocess(batch):
    cAs = []
    cDs = []
    for N in batch:
        channel_data = np.transpose(N)
        cA, cD = pywt.dwt(channel_data, 'coif1')
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
        anchorEEG = data[word][content_idx]

        # the options for positive are any data point with a matching word to the anchor
        # the positive is chosen as the data point that falls circularly after the anchor
        pos_indice = (content_idx + 1) % len(data[word])
        posEEG = data[word][pos_indice]

        # the negative is the next data point (circularly) which has a different word than the anchor
        neg_data_idx = (data_indice + 1) % len(anchors)
        neg_word = list(data.keys())[neg_data_idx]
        neg_contents_idx = np.random.randint(0, len(data[neg_word]))
        negEEG = data[neg_word][neg_contents_idx]

        a, aa = preprocess(anchorEEG)
        p, pp = preprocess(posEEG)
        n, nn = preprocess(negEEG)

        x = tf.stack([*[a, aa], *[p, pp], *[n, nn]], axis=0)
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
    # shuffled_anchors = anchors.copy()
    # np.random.shuffle(shuffled_anchors)
    # anchor=[anchor]
    # shuffled_anchors=[anchor]

    # print(data.keys())
    while True:
        # if True:
        anchor_idx = np.random.randint(0, 100)
        anchor = anchors[anchor_idx]

        word, content_idx, data_indice = anchor[0], anchor[1], anchor[2]
        anchorEEG = data[word][content_idx]

        # the options for positive are any data point with a matching word to the anchor
        # the positive is chosen as the data point that falls circularly after the anchor
        pos_indice = (content_idx + 1) % len(data[word])
        posEEG = data[word][pos_indice]

        # the negative is the next data point (circularly) which has a different word than the anchor
        neg_data_idx = (data_indice + 1) % len(anchors)
        neg_word = list(data.keys())[neg_data_idx]
        neg_contents_idx = np.random.randint(0, len(data[neg_word]))
        negEEG = data[neg_word][neg_contents_idx]

        a, aa = preprocess(anchorEEG)
        p, pp = preprocess(posEEG)
        n, nn = preprocess(negEEG)

        x = tf.stack([*[a, aa], *[p, pp], *[n, nn]], axis=0)
        # x = tf.expand_dims(x, axis=0)

        # print(f"  generator() called!")
        # print(f"    word: ({word}, {content_idx}), pos: ({word}, {pos_indice}), neg: ({neg_word}, {neg_contents_idx})")
        # print(f"    return shape: {x.shape}")
        # print(f"    tf.reduce_mean(x): {tf.reduce_mean(x)}")

        # return x
        yield x


def train_gen(input=None):
    return tf.data.Dataset.from_generator(train_data_generator, output_types=tf.float32)


def validation_gen(input=None):
    return tf.data.Dataset.from_generator(validation_data_generator, output_types=tf.float32)


def main():
    global anchors, data
    os.system("clear")
    # filename = "D:\\NLP Datasets\\word_dict_new.dict"
    # filename = "data/WD/SR_NR2_TSR2_lower_numless_word_dict.dict"
    trainfile = "data/WD/train_dict.dict"
    devfile = "data/WD/dev_dict.dict"
    testfile = "data/WD/test_dict.dict"

    epochs = 10
    batch_size = 15
    loss_margin = 1
    embedding_dim = 100
    MODEL_FILE = "NLP Models\\clustering_model_v001.pt"
    word_embeddings_file = "NLP Models\\word_embeddings_v001.dict"

    # Initialize model
    # model = eeg2vec()  # Pass hyperparams

    # with open(trainfile, 'rb') as datafile:
    #     data = pickle.load(datafile)
    train = data
    # with open(trainfile, 'rb') as datafile:
    #     train = pickle.load(datafile)

    # with open(devfile, 'rb') as datafile:
    #     devdata = pickle.load(datafile)

    # vocab = [word for word in data]

    # Maybe make anchors BPEs??? (to improve vocab size and generalize to the eval analogy vocab)
    vectorized_vocab = {}

    # Sorts the dictionary, but in one line to be 100% safe from memory usage issues
    data = {e[0]: e[1] for e in sorted([[k, v] for k, v in data.items()])}

    anchors = []

    for idx, word in enumerate(data):
        for eeg_idx in range(len(train[word])):
            anchors.append((word, eeg_idx, idx))

    BATCH_SIZE = 6
    # Tensorflow dataset for multiprocess data fetching while the GPU works
    # x = tf.data.Dataset.range(20).map(map_func=generator)
    # x = tf.data.Dataset.range(20)
    # x = x.interleave(map_func=fast_ret, num_parallel_calls=mp.cpu_count())
    # output_shapes=(1, 6, 51, 55))
    # x = tf.data.Dataset.from_generator(generator, output_types=tf.float32).repeat()
    # x = tf.data.Dataset.range(len(anchors))

    x = tf.data.Dataset.range(1)
    x = x.interleave(map_func=train_gen,
                     num_parallel_calls=tf.data.AUTOTUNE,
                     cycle_length=tf.data.AUTOTUNE,
                     block_length=8,
                     deterministic=False)
    x = x.prefetch(tf.data.AUTOTUNE)
    x = x.repeat()  # Random without repeats
    x = x.as_numpy_iterator()

    # y = tf.data.Dataset.range(1)
    # y = y.interleave(map_func=validation_gen,
    #                  num_parallel_calls=tf.data.AUTOTUNE,
    #                  cycle_length=tf.data.AUTOTUNE,
    #                  block_length=8,
    #                  deterministic=False)
    # y = y.prefetch(tf.data.AUTOTUNE)
    # y = y.repeat()  # Random without repeats
    # y = y.as_numpy_iterator()

    print()
    import TFModel
    importlib.reload(TFModel)
    model = TFModel.tf_model()
    model.compile(tfa.optimizers.LAMB())
    # model.fit(x, steps_per_epoch=len(anchors), epochs=1,batch_size=5)
    model.fit(x, steps_per_epoch=1, epochs=1, batch_size=1, )
    model.summary()
    model.fit(x,
              steps_per_epoch=1000, epochs=1,
              batch_size=32,
              # validation_data=y,
              use_multiprocessing=True, workers=mp.cpu_count(),
              callbacks=[TFModel.validation_callback()],
              )
    return

    import TFModel
    importlib.reload(TFModel)
    model = TFModel.tf_model()
    model.compile(tfa.optimizers.LAMB())
    model.fit(new_x)
    # model.build(x.batch(5))
    model.summary()

    return

    print("Saving model...")
    torch.save(model, MODEL_FILE)

    with open(testfile, 'rb') as datafile:
        testdata = pickle.load(datafile)
    # temp = validation_accuracy(model, testdata, embedding_dim)
    temp = 0
    print("Test Accuracy:", temp)
    with torch.no_grad():
        word_embeddings = {}
        for word in data:
            # Find the average of a cluster's embeddings
            word_sum = torch.zeros([dim], dtype=torch.float32)
            for eeg in data[word]:
                cAs, cDs = preprocess([eeg])
                word_sum += model(cAs, cDs)
            word_avg = torch.div(word_sum, len(data[word]))
            word_embeddings[word] = word_avg

    pickle.dump(word_embeddings, open(word_embeddings_file, 'wb'))


main()

if __name__ == "__main__":
    main()
