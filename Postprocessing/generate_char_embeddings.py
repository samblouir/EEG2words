import torch
import numpy as np
from chars2EEGvec import chars2EEGvec
from chars2EEGvec import preprocess
import pickle
import sys

def main():
    if len(sys.argv) != 4:
        print("Usage: generate_char_embeddings.py [Model] [VocabList] [EmbeddingsDictOut]")
        exit()
    model_file = sys.argv[1]
    vocab_file = sys.argv[2]
    out_file = sys.argv[3]
    model = torch.load(model_file)
    with open(vocab_file, 'rb') as f:
        vocab = pickle.load(f)
    with torch.no_grad():
        embeds = {}
        for word in vocab:
            in_word = preprocess(word)
            embeds[word] = model(in_word).numpy()

    with open(out_file, 'wb') as f:
        pickle.dump(embeds, f)
if __name__ == "__main__":
    main()