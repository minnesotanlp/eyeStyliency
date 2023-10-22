import csv
from ia_processing.ia_processing_helpers import Criterion, Accumulator
from matplotlib import pyplot as plt
import os
from statistics import median
from nltk.corpus import stopwords
from flair.models import SequenceTagger
import numpy as np
import nltk
import matplotlib

matplotlib.rcParams.update({'font.size': 20})
accumulators = [Accumulator.SUBTRACTIVE, Accumulator.ALL]
criterions = [Criterion.AVG]
tagger = SequenceTagger.load("flair/pos-english")
SAVE_NAME = '.tmptags'


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


def get_tags(text, imp_words):
    tags = {}
    #sentence = Sentence(text)
    #tagger.predict(sentence)
    #words = sentence.tokenized
    #vals = sentence.to_dict()['all labels']
    # #for tag_dict, w in zip(vals, words):
    #     if w not in stopwords.words('english') and w in imp_words.lower():
    #         tag = tag_dict['value']
    #         if tag not in tags:
    #             tags[tag] = 0
    #         tags[tag] += 1
    words = text.split(' ')
    res = nltk.pos_tag(words)
    for w, pos in res:
        w = w.lower()
        if w not in stopwords.words('english') and w in imp_words.lower():
            if pos not in tags:
                tags[pos] = 0
            if pos == 'IN':
                print(w)
            tags[pos] += 1
    return tags


def tag_file(filename):
    with open(filename) as f:
        reader = csv.reader(f)
        lines = list(reader)
        header = lines[0]
        response_types = header[3:]

        tags = {}
        style_options = [["Negative", "Positive"], ["Polite", "Impolite"]]
        for i, response_type in enumerate(response_types):
            tags[response_type] = {}

        for line in lines[1:]:
            text = line[1]
            words = line[3:]
            for word_list, method in zip(words, response_types):
                method_tags = get_tags(text, word_list)
                for pos, count in method_tags.items():
                    if pos not in tags[method]:
                        tags[method][pos] = 0
                    tags[method][pos] += count

    return tags


def plot(tag_results, ax, crit_name, acc_name):
    category_names = ['IN', 'JJ', 'JJR', 'JJS', 'NN', 'NNP', 'NNS', 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN',
           'VBP', 'VBZ']
    results = {}
    for measure, tag_dict in tag_results.items():
        if measure.endswith('z_score'):
            measure = measure.replace('z_score', '')
        if measure == 'hummingbird annotations':
            measure = 'Human Annotations'
        if measure == 'First Run Dwell':
            measure = 'First Run Dwell Time'
        if measure == 'captum scores':
            measure = 'Integrated Gradients'
        results[measure] = []
        for l in category_names:
            cats = ['JJ', 'RB', 'VB', 'NN', 'PRP']
            abb = ''
            for c in cats:
                if l.startswith(c) and l != c:
                    abb = c
            val = tag_dict[l] if l in tag_dict else 0
            if abb:
                results[measure][-1] += val
            else:
                results[measure].append(val)
        other = 0
        for m in tag_dict:
            if m not in category_names:
                other += tag_dict[m]
        results[measure].append(other)

    for measure in results:
        counts = results[measure]
        new_counts = []
        for i in range(len(counts)):
            new_counts.append(counts[i]/sum(counts))
        results[measure] = new_counts
        print(measure)
        print('sum count', sum(counts))
        print()

    pos_names = []
    cats = ['JJ', 'RB', 'VB', 'NN', 'PRP']
    for p in category_names:
        counts = True
        for c in cats:
            if p.startswith(c) and p != c:
                counts = False
        if counts:
            pos_names.append(p)
    pos_names.append('other')
    plot_categories(results, pos_names, ax)


def get_total_counts(d):
    result = 0
    for k, v in d.items():
        result += v
    return result


def plot_categories(results, category_names, ax):
    """
    Parameters
    ----------
    results : dict
        dict mapping from labels to category values
    category_names : list of str
        The category labels.
    ax: matplob lib ax to use for plotting
    """
    labels = list(results.keys())
    data = np.array(list(results.values()))
    data_cum = data.cumsum(axis=1)
    category_colors = plt.get_cmap('RdYlGn')(np.linspace(0.15, 0.85, data.shape[1]))

    ax.invert_yaxis()
    ax.xaxis.set_visible(False)
    ax.set_xlim(0, np.sum(data, axis=1).max())

    for i, (colname, color) in enumerate(zip(category_names, category_colors)):
        widths = data[:, i]
        starts = data_cum[:, i] - widths
        rects = ax.barh(labels, widths, left=starts, height=0.5,
                        label=colname, color=color)

        r, g, b, _ = color
        text_color = 'white' if r * g * b < 0.3 else 'dimgray'

        ax.bar_label(rects, label_type='center', labels=[f'{x:.0%}' if x > .04 else '' for x in rects.datavalues], color=text_color,
                     fmt='%.0f{}')
    ax.legend(ncol=len(category_names), bbox_to_anchor=(0, 1),
              loc='lower left', fontsize='small')

    return ax


if __name__ == '__main__':
    f, ax = plt.subplots(figsize=(11, 5))
    accumulators = [Accumulator.SUBTRACTIVE]  # accumulators = [Accumulator.SUBTRACTIVE, Accumulator.ALL]
    x_i = 0
    for c in criterions:
        x_i += 1
        foundfile = False
        for a in accumulators:
            filename = 'importantwords/words_{}_{}.csv'.format(c, a)
            if not os.path.exists(filename):
                continue
            foundfile = True
            tags = tag_file(filename)
            keys_to_pop = []
            for response_type in tags:
                if 'z_score' not in response_type.lower() and response_type not in ['hummingbird annotations', 'captum scores']:
                    keys_to_pop.append(response_type)

            for k in keys_to_pop:
                tags.pop(k)
            plot(tags, ax, c, a)
        if not foundfile:
            ax.axis('off')
            continue

    plt.tight_layout()
    plt.show()
