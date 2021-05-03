import numpy as np
from scipy.special import softmax
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import pickle
import sys

def calc_euclidean(x1, x2):
    return np.sum((x1-x2)**2)

def main():
    if len(sys.argv) != 5:
        print("Usage: model-performance.py [OurEmbeddings] [GloveEmbeddings] [RandomEmbeddings] [AnalogyFile]")
        print("Running with default args...")
        ours_file = "FinalCharEmbeddings.dict"
        glove_file = "trimmed_glove_embeddings.dict"
        random_file = "random_embeddings.dict"
        analogy_file = "questions-words.txt"
        # exit()
    else:
        ours_file = sys.argv[1]
        glove_file = sys.argv[2]
        random_file = sys.argv[3]
        analogy_file = sys.argv[4]

    # Gather the analogy questions
    questions_dict = {}
    vocab = set()
    with open(analogy_file, 'r', encoding="utf-8") as f:
        for line in f:
            values = line.split()
            if len(values) == 0:
                continue
            if values[0] == ':':
                section = values[1]
                questions_dict[section] = []
            else:
                values = [value.lower() for value in values]
                questions_dict[section].append(values)
                vocab.update(set(values))

    vocab = list(vocab)

    # Get embeddings from files
    with open(ours_file, 'rb') as f:
        our_embeds = pickle.load(f)
    with open(glove_file, 'rb') as f:
        glove_embeds = pickle.load(f)
    with open(random_file, 'rb') as f:
        random_embeds = pickle.load(f)

    # Set up score structures
    labels = [section for section in questions_dict]
    our_scores = []
    glove_scores = []
    random_scores = []
    print("Sections:")
    for section in questions_dict:
        print('\t', section)
    # Calculate scores for each section
    for section in questions_dict:
        our_score = 0
        glove_score = 0
        random_score = 0
        print("Calculating scores for section:", section)
        # Calculate and average scores
        for analogy in tqdm(questions_dict[section]):
            actual_index = vocab.index(analogy[3])

            our_pred = our_embeds[analogy[1]] - our_embeds[analogy[0]] + our_embeds[analogy[2]]
            our_dists = np.array([calc_euclidean(our_pred, our_embeds[word]) for word in vocab])
            our_score += softmax(-our_dists)[actual_index]

            glove_pred = glove_embeds[analogy[1]] - glove_embeds[analogy[0]] + glove_embeds[analogy[2]]
            glove_dists = np.array([calc_euclidean(glove_pred, glove_embeds[word]) for word in vocab])
            glove_score += softmax(-glove_dists)[actual_index]

            random_pred = random_embeds[analogy[1]] - random_embeds[analogy[0]] + random_embeds[analogy[2]]
            random_dists = np.array([calc_euclidean(random_pred, random_embeds[word]) for word in vocab])
            random_score += softmax(-random_dists)[actual_index]
        # Average and record scores
        our_score /= len(questions_dict[section])
        our_scores.append(our_score)

        glove_score /= len(questions_dict[section])
        glove_scores.append(glove_score)

        random_score /= len(questions_dict[section])
        random_scores.append(random_score)

        print(f"Scores (section: {section}):")
        print('\tOurs:\t', our_score)
        print('\tGlove:\t', glove_score)
        print('\tRandom:\t', random_score)

    print(our_scores, glove_scores, random_scores)
    # Set up bar plot
    # Matplot code from https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html
    x = np.arange(len(labels))  # the label locations
    width = 0.15  # the width of the bars
    softscores = softmax([our_scores, glove_scores, random_scores])
    fig, ax = plt.subplots()
    rects1 = ax.bar(x + width/3, softscores[0], width, label='Ours', align='edge')
    rects2 = ax.bar(x + 2*width/3, softscores[1], width, label='Glove', align='edge')
    rects3 = ax.bar(x + 3*width/3, softscores[2], width, label='Random', align='edge')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Softmax Scores')
    ax.set_title('Scores by Embeddings in Each Section')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)

    fig.tight_layout()

    plt.show()

if __name__ == "__main__":
    main()