import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax
from tqdm import tqdm
import pickle
import sys

labels = ["capital-common-countries",
 "capital-world",
 "currency",
 "city-in-state",
 "family",
 "gram1-adjective-to-adverb",
 "gram2-opposite",
 "gram3-comparative",
 "gram4-superlative",
 "gram5-present-participle",
 "gram6-nationality-adjective",
 "gram7-past-tense",
 "gram8-plural",
 "gram9-plural-verbs"]

our_scores = [0.0011277068060088402, 0.0011216120248049599, 0.0011436999813109361, 0.0011132382449654716, 0.0011526468554336814, 0.0010981989725253024, 0.0010550457019624965, 0.0011676809744903811, 0.0011533618632391918, 0.001125787833161391, 0.0011275413023786103, 0.0011075178897068, 0.001157652988995624, 0.001150372426892663]
glove_scores = [0.7368733300118644, 0.6115903193793514, 0.09013083528494338, 0.11751395841244312, 0.39422096995222206, 0.08111536404535909, 0.08267367024341611, 0.32381551899065314, 0.2814888079361966, 0.14964578002432458, 0.877407380352127, 0.12696211268288082, 0.1551990816200274, 0.27880771204267313]
random_scores = [3.358321351958763e-06, 4.673111574007968e-06, 3.268578763674968e-06, 1.541741611840181e-06, 1.9087878094016046e-06, 3.081952496547868e-06, 1.4544415895213011e-06, 3.829251558967892e-06, 1.9843045661471394e-06, 2.698178046367704e-06, 2.176430960583065e-06, 4.612700033715085e-06, 2.2493188336098928e-06, 7.109939203962873e-06]
# Set up bar plot
# Matplot code from https://matplotlib.org/stable/gallery/lines_bars_and_markers/barchart.html
x = np.arange(len(labels))  # the label locations
width = 0.15  # the width of the bars

fig, ax = plt.subplots()

exp = lambda x: np.e**(x)
log = lambda x: np.log(x)
ax.set_yscale('log')#'function', functions=(exp, log))
rects1 = ax.bar(x + width, our_scores, width, label='Ours', align='edge')
rects2 = ax.bar(x + 2*width, glove_scores, width, label='Glove', align='edge')
rects3 = ax.bar(x + 3*width, random_scores, width, label='Random', align='edge')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Scores')
ax.set_title('Scores by Embeddings in Each Section')
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, fontsize=8)
ax.legend()

ax.bar_label(rects1, padding=3)
ax.bar_label(rects2, padding=3)
ax.bar_label(rects3, padding=3)

plt.gca().set_ylim(bottom=0, top=1)
# fig.tight_layout()

plt.show()