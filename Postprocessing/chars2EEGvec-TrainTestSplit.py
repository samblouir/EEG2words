import random
import pickle
import sys

def main():
    if len(sys.argv) != 4:
        print("Usage: chars2EEGvec-TrainTestSplit.py [EmbeddingDict] [TrainDictOut] [TestDictOut]")
        exit()

    EmbeddingDict = sys.argv[1]
    TrainDictOut = sys.argv[2]
    TestDictOut = sys.argv[3]
    
    with open(EmbeddingDict, 'rb') as embs:
        embeddings = pickle.load(embs)

    wordlist = [word for word in embeddings]
    random.shuffle(wordlist)

    TrainDict = {}
    TestDict = {}

    for word in wordlist[:int(len(wordlist)*0.8)]:
        TrainDict[word] = embeddings[word]

    for word in wordlist[int(len(wordlist)*0.8):]:
        TestDict[word] = embeddings[word]

    with open(TrainDictOut, 'wb') as td:
        pickle.dump(TrainDict, td)

    with open(TestDictOut, 'wb') as td:
        pickle.dump(TestDict, td)

if __name__ == "__main__":
    main()