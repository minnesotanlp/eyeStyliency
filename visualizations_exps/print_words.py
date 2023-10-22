import json
import csv
import math
from ia_processing.ia_processing_helpers import Criterion, Accumulator
from matplotlib import pyplot as plt
import os
from statistics import median
from nltk.corpus import stopwords
from flair.data import Sentence
from flair.models import SequenceTagger
import numpy as np
import json
import nltk

accumulators = [Accumulator.SUBTRACTIVE, Accumulator.ALL]
criterions = [Criterion.AVG, Criterion.TOP_20, Criterion.TOP_33, Criterion.TOP_50]

color_map = {
    'baseline': 'cyan',
    'hummingbird annotations': 'blue',
    'captum scores': 'black',
    'surprisal': 'magenta',
    'Dwell Time': 'red',
    'Reread Time': 'green',
    'Go Past Time': 'purple',
    'First Run Dwell': 'orange',
    'First Fixation Duration': 'brown',
    'Pupil Size': 'gray'
}

marker_map = {'z_score': 'o',
              'raw': 's',
              'non-eye-tracking': 'P'}


def read_file(filename):
    with open(filename) as f:
        word_scores = json.loads(f.read())
    for measure in word_scores:
        if 'run dwell' not in measure.lower() and measure not in ['hummingbird annotations', 'captum scores']:
            continue

        scores = word_scores[measure]
        if len(scores) == 0:
            print()
            continue
        print(measure)
        scores = [s for s in scores if len(s) > 1 and s[2] in ['Impolite']]
        scores = sorted(scores, key=lambda x: x[1])
        scores.reverse()
        print('top 10', scores[:10])
        print()

    print()



if __name__ == '__main__':
    for c in criterions:
        foundfile = False
        for a in accumulators:
            filename = 'wordscores/wordscores_{}_{}.csv'.format(c, a)
            if not os.path.exists(filename):
                continue
            print(filename)
            read_file(filename)
            print()

        #ax.set_xlim([.85, .925])
        #ax.set_ylim([.79, .85])

    #handles, labels = axes[0][0].get_legend_handles_labels()
    #fx = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
    #colors = color_map.values()
    #markers = marker_map.values()
    #handles1 = [fx("s", color) for color in colors]
    #handles2 = [fx(marker, "k") for marker in markers]

    # labels1 = list(color_map.keys())
    # labels2 = list(marker_map.keys())
    # f.supylabel("Average Confidence Score")
    # f.supxlabel("Median Confidence Score")
    # legend1 = plt.legend(handles1, labels1, loc='lower left')
    # legend2 = plt.legend(handles2, labels2, loc='lower right')
    # plt.gca().add_artist(legend1)
    # plt.gca().add_artist(legend2)
    # plt.xticks(rotation=30, horizontalalignment='right')
    # plt.autoscale(True, axis='both', tight=True)
   # plt.show()
