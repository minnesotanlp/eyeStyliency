import json
import csv
import math
from ia_processing.ia_processing_helpers import Criterion, Accumulator
from matplotlib import pyplot as plt
import os
from statistics import median
from scipy import stats
import numpy as np
RESPONSE_FILE = 'openai_results_Criterion.TOP_50_Accumulator.SUBTRACTIVE.csv'

accumulators = [Accumulator.SUBTRACTIVE, Accumulator.ALL]
criterions = [Criterion.AVG]

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

TOP_SCORES = {0: ''}
def top_response(resp, label):
    label = label.lower()
    logprob_dicts = resp["choices"][0]["logprobs"]["top_logprobs"]
    best_res = 0
    for logprobs in logprob_dicts:
        for word, prob in logprobs.items():
            word = word.lower().strip()
            if label in word or label[0:3] == word:
                prob = math.exp(prob)
                if prob > best_res:
                    best_res = prob
    return best_res


def score_response(resp, label):
    label = label.lower()
    logprob_dicts = resp["choices"][0]["logprobs"]["top_logprobs"]
    total = 0
    for logprobs in logprob_dicts:
        for word, prob in logprobs.items():
            word = word.lower().strip()
            if label in word or label[0:3] == word:
                total += math.exp(prob)
    return total


def correct_response(resp, correct_label, incorrect_label):
    correct_prob = score_response(resp, correct_label)
    incorrect_prob = score_response(resp, incorrect_label)
    return correct_prob > incorrect_prob

# can have dot for SUBTRACTIVE, * for all
# can have different color for each base
# y axis can be confidence for correct answer, x axis can be confidence for incorrect answer?


def score_file(filename):
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
            score_results[response_type] = []
            accuracy_results[response_type] = []
            opposite_score_results[response_type] = []

        for line in lines[2:]:
            stim_id = line[0]
            text = line[1]
            style = line[2]
            # if style in ["Negative", "Positive"]:
            #     continue
            options = style_options[0] if style in style_options else style_options[1]
            incorrect = options[0] if style != options[0] else options[1]
            responses = line[3:]

            all_methods = True
            for i, response in enumerate(responses):
                if len(response) == 0 or response == "NA":
                    print(response_types[i])
                    all_methods = False

            if not all_methods:
                continue

            for response, method in zip(responses, response_types):
                if len(response) == 0 or response == "NA":
                    continue
                response_dict = json.loads(response)
                score_results[method].append(top_response(response_dict, style))
                score = top_response(response_dict, style)
                tenth_best = sorted(TOP_SCORES.keys())[0]


                opposite_score_results[method].append(score_response(response_dict, incorrect))
                accuracy_results[method].append(1 if correct_response(response_dict, style, incorrect) else 0)

            hb_score = score_results['hummingbird annotations response'][-1]
            # for method in score_results:
            #     if score_results[method][-1] > hb_score:
            #         print('{} beats hb by {}'.format(method, score_results[method][-1] - hb_score))
            # print('stim id', stim_id)
            # print()
        return opposite_score_results, score_results


def plot_agg(score_results1, score_results2, ax):
    pass


def plot_eye(opposite_results, score_results, ax):
    eye_colors = ['midnightblue', 'darkslateblue', 'blue', 'slateblue', 'royalblue', 'cornflowerblue', 'dodgerblue', 'lightskyblue',
                  'darkturquoise', 'cyan', 'aquamarine', 'cadetblue', 'steelblue', 'mediumslateblue']
    curr_eye = 0
    non_eye_colors = ['orange', 'gold', 'brown', 'darkgoldenrod']
    curr_non_eye = 0
    x = []
    y = []
    errors = []
    x_ = 0
    for method, num in score_results.items():
        num = [n for n in num if n > 0.5]
        if len(num) == 0:
            score = 0
        else:
            score = sum(num)/len(num)
        #print(len(num))
        ci = stats.norm.interval(alpha=0.95, loc=stats.gmean(num), scale=np.std(num))
        op_num = opposite_results[method]
        op_score = median(num)
        #op_score = sum(op_num)/len(op_num)
        if method == 'baseline':
            method_name = method
            color = non_eye_colors[curr_non_eye]
            curr_non_eye += 1
            agg_type = 'non-eye-tracking'
        elif method in ['hummingbird annotations response', 'captum scores response', 'surprisal response']:
            components = method.split(' ')
            method_name = ' '.join(components[:-1])
            color = non_eye_colors[curr_non_eye]
            curr_non_eye += 1
            agg_type = 'non-eye-tracking'
            #if method.startswith('hummingbird'):
            #    score, op_score = 0.8280430306835632, 0.8955129357063171
        else:
            components = method.split(' ')[:-1]  # last component is just result
            method_name, agg_type = ' '.join(components[:-1]), components[-1]
            if agg_type in ['pred', 'z_score']:
                continue
            color = eye_colors[curr_eye]
            curr_eye += 2
        m = marker_map[agg_type]
        ax.bar(x_, score, 1, color=color, label=method_name)
        x_ += 1
        #ax.scatter([op_score], [score], c=color, marker=m, label=method_name)
        x.append(op_score)
        y.append(score)
        print(np.std(num))
        errors.append(2 * np.std(num))
    return x, y, errors



def plot(opposite_results, score_results, ax, crit_name, acc_name):
    for method, num in score_results.items():
        num = [num for num in num if num > 0.5]
        if len(num) == 0:
            score = 0
            num = [0]
        else:
            score = sum(num)/len(num)
            score2 = sum([1 for n in num if n > 0.5])
        op_num = opposite_results[method]
        op_score = median(num)
        #op_score = sum(op_num)/len(op_num)
        if method == 'baseline':
            method_name = method
            agg_type = 'non-eye-tracking'
        elif method in ['hummingbird annotations response', 'captum scores response', 'surprisal response']:
            components = method.split(' ')
            method_name = ' '.join(components[:-1])
            agg_type = 'non-eye-tracking'
            #if method.startswith('hummingbird'):
            #    score, op_score = 0.8280430306835632, 0.8955129357063171
        else:
            components = method.split(' ')[:-1]  # last component is just result
            method_name, agg_type = ' '.join(components[:-1]), components[-1]
            if agg_type == 'pred':
                agg_type = 'raw'
        c = color_map[method_name]
        m = marker_map[agg_type]
        print(f"'{method}': {score2},")
        ax.scatter([op_score], [score], c=c, marker=m, label=method_name)


if __name__ != '__main__':
    f, ax = plt.subplots(figsize=(9, 7))
    accumulator = Accumulator.SUBTRACTIVE
    criterion = Criterion.AVG
    filename = 'openai_results_{}2_{}.csv'.format(criterion, accumulator)
    x, y = score_file(filename)
    print(y.items())
    f.supxlabel('Median Confidence Score')
    f.supylabel('Mean Confidence Score')
    op_score, score, err = plot_eye(x, y, ax)
    plt.errorbar(list(range(len(op_score))), op_score, yerr=err, fmt='none', color='black')
    ax.set_ylim([.2,1.0])
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    f, axes = plt.subplots(2, 2)
    accumulators = [Accumulator.ALL, Accumulator.SUBTRACTIVE]
    x_i = 0
    y_i = 0
    for c in criterions:
        ax = axes[x_i % 2][y_i % 2]
        y_i += 1
        x_i += y_i % 2
        print(x_i, y_i)
        ax.grid()
        foundfile = False
        for a in accumulators:
            filename = 'openai_results_{}_{}.csv'.format(c, a)
            if not os.path.exists(filename):
                continue
            foundfile = True
            print('!!!!!!!!!!!\n')
            print(filename)
            x, y = score_file(filename)
            plot(x, y, ax, c, a)
        if not foundfile:
            ax.axis('off')
            continue
        ax.set_title(c, fontsize='small', loc='left')
        #ax.set_xlim([.77, .82])
        #ax.set_ylim([.72, .79])

    #handles, labels = axes[0][0].get_legend_handles_labels()
    fx = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
    colors = color_map.values()
    markers = marker_map.values()
    handles1 = [fx("s", color) for color in colors]
    handles2 = [fx(marker, "k") for marker in markers]

    labels1 = list(color_map.keys())
    labels2 = list(marker_map.keys())
    f.supylabel("Average Confidence Score")
    f.supxlabel("Median Confidence Score")
    legend1 = plt.legend(handles1, labels1, loc='lower left')
    legend2 = plt.legend(handles2, labels2, loc='lower right')
    plt.gca().add_artist(legend1)
    plt.gca().add_artist(legend2)
    # plt.xticks(rotation=30, horizontalalignment='right')
    plt.autoscale(True, axis='both', tight=True)
    print(TOP_SCORES)
    plt.show()
