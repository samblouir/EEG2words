import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import sys

torch.manual_seed(42)

class chars2EEGvec(nn.Module):
    # Class that defines our model
    def __init__(self, embedding_dim, hidden_dim, vocab_size, EEGvec_dim, num_layers):
        super(chars2EEGvec, self).__init__()
        self.EEGvec_dim = EEGvec_dim

        self.char_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # The LSTM takes character embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers)

        # A second LSTM which learns the data in reverse.
        self.rev_lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2EEGvec = nn.Linear(hidden_dim * 2, EEGvec_dim) #hidden_dim * 2, EEGvec_dim

    # This is the forward computation, which constructs the computation graph
    def forward(self, word):
        
        # Get the embeddings
        embeds = self.char_embeddings(word)
        # put them through the LSTM and get its output
        lstm_out, _ = self.lstm(embeds.view(len(word), 1, -1))
        # put them through the second LSTM backwards and get its output
        rev_lstm_out, _ = self.lstm(torch.flip(embeds.view(len(word), 1, -1), [0, 1]))
        rev_lstm_out = torch.flip(rev_lstm_out.view(len(word), 1, -1), [0, 1])
        # Concatenate LSTM outputs into biLSTM
        bilstm_out = torch.cat((lstm_out.view(len(word), 1, -1), rev_lstm_out.view(len(word), 1, -1)),1)

        # pass those biLSTM outputs through the linear layer
        EEGvec = self.hidden2EEGvec(bilstm_out.view(len(word), -1)).view(self.EEGvec_dim, -1).sum(1)

        return EEGvec

class Euclidean_Distance_Loss(nn.Module):
    def __init__(self):
        super(Euclidean_Distance_Loss, self).__init__()
        
    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return 1 - np.e**(-calc_euclidean(y_true, y_pred))


# Acquired from https://www.kaggle.com/hirotaka0122/triplet-loss-with-pytorch
def calc_euclidean(x1, x2):
    return (x1 - x2).pow(2).sum(0)

def validation_accuracy(model, TestVecs):
    with torch.no_grad():
        dist = 0
        # Calculate the average dist
        for word in TestVecs:
            ids = preprocess(word)
            output = model(ids)
            target = torch.Tensor(TestVecs[word])
            dist += calc_euclidean(output, target).tolist()
        dist /= len(TestVecs)
        score = np.e**(-dist)
    return score

# Turns a word into a tensor of ids for each character
def preprocess(word):
    # Weird characters are put as index 26 while a-z are 0-25
    ids = [((ord(char)-ord('a')) if (ord(char)-ord('a')) in range(26) else 26) for char in word.lower()]
    return torch.tensor(ids, dtype=torch.long)


def train(TrainFile, TestFile, ModelFile):
    # Hyperparameters
    EMBEDDING_DIM = 125
    HIDDEN_DIM = 200
    LAYERS = 2
    PATIENCE = 10
    EPOCHS = 100
    EEGVEC_DIM = 100

    # Load the data
    with open(TrainFile, 'rb') as embs:
        TrainVecs = pickle.load(embs)
    with open(TestFile, 'rb') as embs:
        TestVecs = pickle.load(embs)

    # Initialize the model
    model = chars2EEGvec(embedding_dim=EMBEDDING_DIM,
                        hidden_dim=HIDDEN_DIM,
                        vocab_size=27,
                        EEGvec_dim=EEGVEC_DIM,
                        num_layers=LAYERS)
    
    # Loss function to use
    loss_function = Euclidean_Distance_Loss()
    # loss_function = nn.CosineEmbeddingLoss()

    # Optimizer to use during training
    optimizer = optim.Adam(model.parameters())

    # For early stopping
    patience_left = PATIENCE
    max_validation = 0

    # Training loop
    for epoch in range(EPOCHS):
        print(f"Starting epoch {epoch}...")
        avg_loss = 0
        for word in tqdm(TrainVecs):
            # Clear the gradients
            model.zero_grad()

            # Prepare a word
            word_in = preprocess(word)

            # Generate output and target
            target = torch.tensor(TrainVecs[word], dtype=torch.float32)
            output = model(word_in)

            # Calculate the loss
           # y = torch.tensor(output.size(1))#output.size(1), dtype=torch.int32).fill_(1)
            loss = loss_function(output, target)
            avg_loss += loss.tolist()

            loss.backward()
            optimizer.step()

        # Average loss over the epoch
        avg_loss = avg_loss/len(TrainVecs)
        print("Average Loss:", format(avg_loss, '.4f'))

        # The current accuracy on the testing set
        validation = validation_accuracy(model, TestVecs)
        print("Validation Accuracy:", format(validation, '.4f'))

        if validation > max_validation:
            max_validation = validation
            patience_left = PATIENCE
            print("New maximum, saving model to file \'"+ModelFile+"\'")
            torch.save(model, ModelFile)
        else:
            patience_left -= 1
            if patience_left == 0:
                print("Patience ran out... Ending training at epoch " + str(epoch) + "...")
                print("-------------------------------------------")
                break
        print("-------------------------------------------")

def main():
    if len(sys.argv) != 4:
        print("Usage: chars2EEGvec.py [TrainDict] [TestDict] [ModelOut]")
        exit()

    TrainDict = sys.argv[1]
    TestDict = sys.argv[2]
    ModelOut = sys.argv[3]

    train(TrainFile=TrainDict,
        TestFile=TestDict,
        ModelFile=ModelOut)

if __name__ == "__main__":
    main()