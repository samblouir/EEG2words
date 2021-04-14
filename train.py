import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import pywt as wt
import pickle
import numpy as np

torch.manual_seed(42)

dim = 50


class eeg2vec(nn.Module):
    # Class that defines our model
    def __init__(self, batch_size, channel_indxs, time_window):
        super(eeg2vec, self).__init__()
        self.channel_indxs = channel_indxs
        self.batch_size = batch_size
        # self.hidden_dim = hidden_dim
        self.cAconv1 = nn.Conv1d(len(channel_indxs), 38, 5)
        self.cDconv1 = nn.Conv1d(len(channel_indxs), 38, 5)
        #
        # self.cAmaxpool1 = nn.MaxPool1d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        # self.cDmaxpool1 = nn.MaxPool1d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
        #
        # self.cAconv2 = nn.Conv1d(len(channel_indxs), hidden_dim)
        # self.cDconv2 = nn.Conv1d(channels, hidden_dim)
        #
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
    def forward(self, cAs, cDs):
        cAouts1 = self.cAconv1(cAs)
        cDouts1 = self.cAconv1(cDs)
        '''
        # Get the embeddings
        embeds = self.word_embeddings(sentence)
        # put them through the LSTM and get its output
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        # pass those outputs through the linear layer
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        # convert the logits to a log probability distribution
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores
        '''


# Acquired from https://www.kaggle.com/hirotaka0122/triplet-loss-with-pytorch
def calc_euclidean(x1, x2):
    return (x1 - x2).pow(2).sum(1)


class TripletLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, negative: torch.Tensor) -> torch.Tensor:
        distance_positive = self.calc_euclidean(anchor, positive)
        distance_negative = self.calc_euclidean(anchor, negative)
        losses = torch.relu(distance_positive - distance_negative + self.margin)

        return losses.mean()


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
    return torch.tensor(cAs, dtype=torch.float32), torch.tensor(cDs, dtype=torch.float32)


def main():
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
    model = eeg2vec()  # Pass hyperparams

    # Loss function to use during training
    loss_function = TripletLoss(loss_margin)

    # Optimizer to use during training
    optimizer = optim.Adam(model.parameters())

    with open(trainfile, 'rb') as datafile:
        data = pickle.load(datafile)

    train = data
    # with open(trainfile, 'rb') as datafile:
    #     train = pickle.load(datafile)

    with open(devfile, 'rb') as datafile:
        devdata = pickle.load(datafile)

    vocab = [word for word in data]

    # Maybe make anchors BPEs??? (to improve vocab size and generalize to the eval analogy vocab)
    anchors = []
    for word in data:
        for eeg_idx in range(len(train[word])):
            anchors.append((word, eeg_idx))

    for epoch in range(epochs):
        avg_loss = 0

        print(f"Starting epoch {epoch}...")
        shuffled_anchors = anchors.copy()
        np.random.shuffle(shuffled_anchors)

        batch = {}
        model.zero_grad()
        for anc_idx, anchor in enumerate(shuffled_anchors):
            anchorEEG = data[anchor[0]][anchor[1]]

            # the options for positive are any data point witha  matching word to the anchor
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
            batch['anchor'].append(anchorEEG)
            batch['pos'].append(posEEG)
            batch['neg'].append(negEEG)

            # pass the batch to the model after a full batch is collected or we are at the end of the data
            if len(batch['anchor']) == batch_size or anc_idx == len(shuffled_anchors) - 1:
                cAs, cDs = preprocess(batch['anchor'])
                anchor_outs = model(cAs, cDs)

                cAs, cDs = preprocess(batch['pos'])
                pos_outs = model(cAs, cDs)

                cAs, cDs = preprocess(batch['neg'])
                neg_outs = model(cAs, cDs)

                # Compute loss and backprop for each triplet in 
                for batchnum in range(len(batch['anchor'])):
                    loss = loss_function(anchor=anchor_outs[batchnum],
                                         positive=pos_outs[batchnum],
                                         negative=neg_outs[batchnum])
                    avg_loss += loss.tolist()
                    loss.backward()
                    optimizer.step()

                batch = []
                model.zero_grad()
        # Average loss over the epoch
        avg_loss /= len(anchors)
        print("Average Loss:", format(avg_loss, '.4f'))
        # The current quality of the clustering
        # validation = validation_accuracy(model, devdata, embedding_dim)
        validation = 0
        print("Validation Accuracy:", format(validation, '.4f'))
        print("-------------------------------------------")
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


if __name__ == "__main__":
    main()
