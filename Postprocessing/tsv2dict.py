import sys
import numpy as np
import pickle

def main():
    if len(sys.argv) != 4:
        print("Usage: train [Labels] [Vectors] [DictOut]")
        exit()

    labelfile = sys.argv[1]
    vectorfile = sys.argv[2]
    outfile = sys.argv[3]
    with open(labelfile, 'rt') as lf:
        lines = lf.readlines()
        labels = [label.strip() for label in lines]

    with open(vectorfile, 'rt') as vf:
        lines = vf.readlines()
        vectors = []
        for line in lines:
            vectors.append(np.array([value.strip() for value in line.split('\t')], dtype=np.float32))

    embedding_dict = {label:vector for label, vector in zip(labels, vectors)}
    with open(outfile, 'wb') as of:
        pickle.dump(embedding_dict, of)

if __name__ == "__main__":
    main()