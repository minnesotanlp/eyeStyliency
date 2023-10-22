import json
import csv
import math
from ia_processing.ia_processing_helpers import Criterion, Accumulator
from matplotlib import pyplot as plt
import os
from statistics import median, stdev
from scipy import stats
from matplotlib_venn import venn3, venn3_circles
import numpy as np
import ast
from utils.utils import *
import matplotlib
import textwrap
import matplotlib.patches as mpatches

matplotlib.rcParams.update({'font.size': 20})
matplotlib.use('TkAgg')

accumulators = [Accumulator.SUBTRACTIVE]
shots = [1, 2, 4]
color_map = {
    'baseline': 'cyan',
    'hummingbird annotations': 'red',
    'captum scores': 'orange',
    'surprisal': 'gold',
    'Dwell Time': 'blue',
    'Reread Time': 'midnightblue',
    'Go Past Time': 'darkturquoise',
    'First Run Dwell': 'deepskyblue',
    'First Fixation Duration': 'cornflowerblue',
    'Pupil Size': 'mediumslateblue'
}

marker_map = {'z_score': 'o',
              'raw': 's',
              'non-eye-tracking': 'P'}

important_words = {}


# from stackoverflow: https://stackoverflow.com/questions/64055112/how-to-make-a-multi-column-text-annotation-in-matplotlib
def place_column_text(ax, text, xy, wrap_n, shift, bbox=False, **kwargs):
    """ Creates a text annotation with the text in columns.
    The text columns are provided by a list of strings. A surrounding box can be added via bbox=True parameter.
    If so, FancyBboxPatch kwargs can be specified.

    The width of the column can be specified by wrap_n, the shift parameter determines how far apart the columns are.
    The axes are specified by the ax parameter.
    """
    # place the individual text boxes, with a bbox to extract details from later
    x, y = xy
    n = 0
    text_boxes = []
    xs = []
    for i in text:
        text = '\n'.join(i)
        box = ax.text(x=x + n, y=y, s=text, va='top', ha='left',
                      bbox=dict(alpha=0, boxstyle='square,pad=0'))
        #box.set_bbox(dict(facecolor='gray', alpha=0.5, edgecolor='gray'))
        text_boxes.append(box)
        xs.append(x + n)
        n += shift
    #ax.add_patch(mpatches.FancyBboxPatch(xy=(x + n/3, y-0.25), width=0.1, height=0.05, facecolor='gray', alpha=0.5))


def score_file(filename, target_style='Impolite'):
    with open(filename) as f:
        reader = csv.reader(f)
        lines = list(reader)
        header = lines[0]
        response_types = header[3:]
        print(response_types)

        score_results = {}
        opposite_score_results = {}
        accuracy_results = {}
        style_options = [["Negative", "Positive"], ["Polite", "Impolite"]]
        for response_type in response_types:
            if 'important words' not in response_type:
                continue
            print(response_type)
            if 'Dwell' not in response_type and 'hummingbird' not in response_type and 'captum' not in response_type:
                continue
            important_words[response_type] = []
            score_results[response_type] = []
            accuracy_results[response_type] = []
            opposite_score_results[response_type] = []
        for line in lines[1:]:
            stim_id = line[0]
            text = line[1]
            style = line[2]
            if style != target_style:
                continue
            if int(stim_id) in [79, 80, 81, 82, 83]:
                print('ahahaha')
                continue
            options = style_options[0] if style in style_options else style_options[1]
            responses = line[3:]

            for response_list, method in zip(responses, response_types):
                if 'important words' not in method:
                    continue
                if 'Dwell' not in method and 'hummingbird' not in method and 'captum' not in method:
                    continue
                try:
                   response_list = ast.literal_eval(response_list)
                   important_words[method] += response_list
                except:
                    print('bad eval', response_list)

    hb = set(important_words['hummingbird annotations important words'])
    capt = set(important_words['captum scores important words'])
    print('hb len', len(hb))
    print('capt len', len(capt))
    keys = list(important_words.keys())[2:]
    for key in keys:
        s = important_words[key]
        s = set(s)
        print(key)
        print(len(s))
        print('intersection', set.intersection(s, hb, capt))
        print('only cap', capt - hb - s)
        print('only hb', hb - capt - s)
        print('only eye', s - hb - capt)
        print('hb + s', set.union(hb, s) - capt)
        print('s + capt', set.union(s, capt) - hb)
        print()

def label_columns(wordset, ax, pos):
    col_len = 15
    wordlist = list(wordset)
    for w in wordlist:
        if 'verify' in w:
            print(wordlist)
    wordlist = [w for w in wordlist if len(w) > 5]
    wordlist = [' '.join(w.split(' ')[2:]) if len(w) > col_len else w for w in wordlist]
    half_idx = int(len(wordlist) / 2)
    col_text = [wordlist[:half_idx], wordlist[half_idx:]]
    place_column_text(ax, col_text, pos, col_len, 0.15)

def do_print(wordset, scores):
    wordset = sorted(list(wordset), key=lambda x: scores.index(x))
    print(wordset[:10])

def do_multi_print(wordset, scoreslist):
    wordset = sorted(list(wordset), key=lambda x: sum([s.index(x) for s in scoreslist]))
    print(wordset)

def plot_scoresfile(filename, target_style='Impolite'):
    with open(filename) as f:
        scores_dict = json.load(f)
        important_words = {}
        for key, scores in scores_dict.items():
            if 'Dwell' not in key and 'hummingbird' not in key and 'captum' not in key:
                continue
            scores = flatten(scores)
            scores = [(word.replace('¬†', '').replace('"', ''), score, style) for word, score, style in scores]
            scores = sorted(scores, key=lambda x: x[1])
            scores.reverse()
            important_words[key] = [word for word, score, _ in scores if score > 0]
        A = set(important_words['hummingbird annotations'])
        B = set(important_words['captum scores'])
        C = set(important_words['Dwell Time z_score'])
        f, ax = plt.subplots(figsize=(11, 8))
        v = venn3([A, B, C], ('Human Annotations', 'Integrated Gradients', 'Dwell Time'))

        print('dt')
        do_print(C - A - B, important_words['Dwell Time z_score'])
        print('\ncaptum only', B - A - C)
        do_print(B - A - C, important_words['captum scores'])
        print('\nhumming only', A - B - C)
        do_print(A - B - C, important_words['hummingbird annotations'])

        print('\ncap dwell')
        do_multi_print(C & B - A, [important_words['captum scores'], important_words['Dwell Time z_score']])

        print('\nhumming + cap')
        do_multi_print(B & A - C, [important_words['captum scores'], important_words['hummingbird annotations']])

        print('\ndwell + humm', A & C - B)
        do_multi_print(C & A - B, [important_words['Dwell Time z_score'], important_words['hummingbird annotations']])

        print('\nall three', A & B & C)
        plt.tight_layout()
        plt.savefig('Venn diagram.pdf')



if __name__ == '__main__':
   #filename = 'openai_1shot_Criterion.AVG_Accumulator.SUBTRACTIVE.csv'
   #score_file(filename, target_style='Polite')
   plot_scoresfile('wordscores/wordscores_Criterion.TOP_50_Accumulator.SUBTRACTIVE.csv', target_style='Polite')
   #plt.figure(figsize=(20, 30))