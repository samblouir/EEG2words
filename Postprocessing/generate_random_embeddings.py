import numpy as np
import pickle
import sys

def main():
    if len(sys.argv) != 3:
        print("Usage: generate_random_embeddings.py [VocabList] [EmbeddingsDictOut]")
        exit()
    vocab_file = sys.argv[1]
    out_file = sys.argv[2]
    with open(vocab_file, 'rb') as f:
        vocab = pickle.load(f)

    embeds = {}
    for word in vocab:
        embeds[word] = np.random.rand(100)

    with open(out_file, 'wb') as f:
        pickle.dump(embeds, f)

if __name__ == "__main__":
    main()