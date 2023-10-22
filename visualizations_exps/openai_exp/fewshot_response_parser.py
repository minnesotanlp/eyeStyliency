import json
import csv
import math
from ia_processing.ia_processing_helpers import Criterion, Accumulator
from matplotlib import pyplot as plt
import os
from statistics import median, stdev
from scipy import stats
import numpy as np
import ast
from utils.utils import *
import matplotlib
import matplotlib.patches as mpatches

matplotlib.rcParams.update({'font.size': 20})
RESPONSE_FILE = 'openai_results_Criterion.TOP_50_Accumulator.SUBTRACTIVE.csv'

METHODS_TO_SHOW = ['Baseline', 'Human Annotations', 'Integrated Gradients', 'hummingbird annotations', 'Dwell Time', 'Reread Time', 'First Fixation Duration', 'Hybrid Score', 'captum scores']

accumulators = [Accumulator.SUBTRACTIVE, Accumulator.ALL]
#criterions = [Criterion.AVG, Criterion.EXP, Criterion.TOP_33, Criterion.TOP_50]
shots = [0, 1, 2, 4]
color_map = {
    'Baseline': 'gray',
    'Human Annotations': 'red',
    'Integrated Gradients': 'green',
    'surprisal': 'gold',
    'Dwell Time': 'deepskyblue',
    'Reread Time': 'blue',
    'Go Past Time': 'darkturquoise',
    'First Run Dwell': 'deepskyblue',
    'First Fixation Duration': 'cornflowerblue',
    'Pupil Size': 'mediumslateblue',
    'Hybrid Score': 'black'
}

hatch_map = {
    'Baseline': '',
    'Human Annotations': '--',
    'Integrated Gradients': 'OO',
    'surprisal': 'gold',
    'Dwell Time': '**',
    'Reread Time': 'xx',
    'Go Past Time': 'oo',
    'First Run Dwell': '||',
    'First Fixation Duration': '..',
    'Pupil Size': 'mediumslateblue',
    'Hybrid Score': '//'
}

marker_map = {'z_score': 'o',
              'raw': 's',
              'non-eye-tracking': 'P'}

line_map = {'z_score': 'solid',
            'raw': 'solid',
            'non-eye-tracking': 'solid',
            'hybrid': 'dashdot',
            'pred': 'solid'}

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
    correct_log = top_response(resp, correct_label)
    incorrect_log = top_response(resp, incorrect_label)
    scores = [correct_log, incorrect_log]
    e_x = np.exp(scores - np.max(scores))
    softmax_correct, softmax_incorrect = e_x / e_x.sum()
    return softmax_correct > softmax_incorrect

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
            if 'important words' in response_type or response_type == 'baseline':
                continue
            score_results[response_type] = []
            accuracy_results[response_type] = []
            opposite_score_results[response_type] = []
        tot = {method: 0 for method in score_results}
        nah = {method: 0 for method in score_results}
        for line in lines[1:]:
            stim_id = line[0]
            text = line[1]
            style = line[2]
            # if style != 'Polite':
            #     continue
            options = style_options[0] if style in style_options else style_options[1]
            incorrect = options[0] if style != options[0] else options[1]
            responses = line[3:]

            all_methods = True
            for response_list, method in zip(responses, response_types):
                if 'important words' in method or method == 'baseline':
                    continue
                if len(response_list) == 0 or response_list in ['NA', '']:
                    all_methods = False

            if not all_methods:
                continue
            print(response_types)

            for response_list, method in zip(responses, response_types):
                print(method)
                if 'important words' in method or method == 'baseline' or method == 'baseline score':
                    continue
                if len(response_list) == 0 or response_list in ['NA', '']:
                    continue
                response_list = ast.literal_eval(response_list)
                response_scores = []
                accuracy_scores = []
                for r in response_list:
                    print(r)
                    response_dict = json.loads(r)
                    response_scores.append(top_response(response_dict, style))
                    accuracy_scores.append(correct_response(response_dict, style, incorrect))
                score_results[method].append(response_scores)
                accuracy_results[method].append(accuracy_scores)
                score = top_response(response_dict, style)
                tenth_best = sorted(TOP_SCORES.keys())[0]


                opposite_score_results[method].append(score_response(response_dict, incorrect))

            #hb_score = score_results['hummingbird annotations response'][-1]
            #for method in score_results:
            #    if score_results[method][-1] > hb_score:
            #        tot[method] += 1
                    #print('{} beats hb by {}'.format(method, score_results[method][-1] - hb_score))
            #    elif not method.startswith('humm'):
            #        nah[method] += 1
        # for method in tot:
        #     if method.startswith('humm') or nah[method] + tot[method] == 0:
        #         continue
            #print(f'{method}: percentage doing better is {tot[method]/(nah[method] + tot[method])}')
        #print('stim id', stim_id)
        print()
        return accuracy_results, score_results


def bar_plot(score_results_list, ax):
    n_shots = [0, 1, 2, 4]
    scores = {method: [] for method in score_results_list[0].keys()}
    errs = {method: [] for method in score_results_list[0].keys()}
    for score_results in score_results_list:
        for method, num in score_results.items():
            totals = []
            for i in range(len(num[0])):
                sc = [1 if n[i] else 0 for n in num]
                totals.append(sum(sc)/len(sc))
            errs[method].append(2*np.nanstd(totals) * 100)
            l = len((num))
            num = [1 for n in num if n.count(True) > 3 ]
            scores[method].append(sum(num)/l * 100)

    group_num = 0  # for bar plotting

    for method in score_results:
        if len(scores[method]) != len(n_shots):
            print('ok', method)
            continue
        if method in ['hummingbird annotations response', 'captum scores response', 'surprisal response', 'hybrid score', 'baseline']:
            components = method.split(' ')
            method_name = ' '.join(components[:-1])
            agg_type = 'non-eye-tracking'
            # if method.startswith('hummingbird'):
            #    score, op_score = 0.8280430306835632, 0.8955129357063171
            if method in ['hybrid score', 'baseline']:
                method_name = method.title()
                agg_type = 'hybrid'
            if method in ['hummingbird annotations response']:
                method_name = 'Human Annotations'
            if method_name == 'captum scores':
                method_name = 'Integrated Gradients'
        else:
            components = method.split(' ')[:-1]  # last component is just result
            method_name, agg_type = ' '.join(components[:-1]), components[-1]
            if agg_type in ['raw', 'pred']:
                continue
         #print(method, scores[method])
        if method_name not in METHODS_TO_SHOW:
            continue
        n_shots = np.asarray([0, 1, 2, 3])
        #ax.plot(n_shots, scores[method], color=color_map[method_name], linestyle=line_map[agg_type], label=method_name)
        ax.bar(np.asarray(n_shots) - 0.1*group_num, scores[method], width=0.1, color=color_map[method_name], label=method_name,
               alpha=0.5, edgecolor=color_map[method_name], hatch=hatch_map[method_name])
        ax.set_xlabel('N Shots')
        ax.set_ylabel('Accuracy (%)')
        ax.set_xticks(n_shots - 0.1 * 2)
        ax.set_xticklabels([0, 1, 2, 4])
        ax.set_ylim([80, 100])
        #ax.set_yticks([375, 380, 385, 390])
        #ax.set_ylim([370, 395])
        #ax.fill_between(n_shots, [s - e for s, e in zip(scores[method], errs[method])],
        #                [s + e for s, e in zip(scores[method], errs[method])], color=color_map[method_name], alpha=0.5)
        #
        ax.errorbar(np.asarray(n_shots) - 0.1*group_num, scores[method], yerr=errs[method], color=color_map[method_name], capsize=3, linestyle='none')
        group_num += 1
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(handles[::-1], labels[::-1], framealpha=1)


def plot(score_results_list, accuracy_list, ax):
    eye_colors = ['midnightblue', 'darkslateblue', 'blue', 'slateblue', 'royalblue', 'cornflowerblue', 'dodgerblue', 'lightskyblue',
                  'darkturquoise', 'cyan', 'aquamarine', 'cadetblue', 'steelblue', 'mediumslateblue']
    curr_eye = 0
    non_eye_colors = ['orange', 'gold', 'brown', 'darkgoldenrod', 'black']
    curr_non_eye = 0
    n_shots = [0, 1, 2, 4]
    scores = {method: [] for method in score_results_list[0].keys()}
    errs = {method: [] for method in score_results_list[0].keys()}
    for score_results, accuracy_results in zip(score_results_list, accuracy_list):
        for method, num in score_results.items():
            accs = accuracy_results[method]
            accs = [1 if a.count(True) > 2 else 0 for a in accs]
            totals = []
            for i in range(len(num[0])):
                sc = [n[i] for n in num if accs[i] > 0]
                totals.append(sum(sc)/len(sc))
            #num = flatten(num)
            if len(num) == 0:
                continue
            num = [sum(n)/len(n) for i, n in enumerate(num) if accs[i] > 0]
            score = sum(num)/len(num)
            scores[method].append(score)
            errs[method].append(2*np.nanstd(totals))

    group_num = 0
    for method in score_results:
        if method in ['hummingbird annotations response', 'captum scores response', 'surprisal response', 'hybrid score', 'baseline']:
            components = method.split(' ')
            method_name = ' '.join(components[:-1])
            color = non_eye_colors[curr_non_eye]
            curr_non_eye += 1
            agg_type = 'non-eye-tracking'
            # if method.startswith('hummingbird'):
            #    score, op_score = 0.8280430306835632, 0.8955129357063171
            if method in ['hybrid score', 'baseline']:
                method_name = method.title()
                agg_type = 'hybrid'
            if method in ['hummingbird annotations response']:
                method_name = 'Human Annotations'
            if method_name == 'captum scores':
                method_name = 'Integrated Gradients'
        else:
            components = method.split(' ')[:-1]  # last component is just result
            method_name, agg_type = ' '.join(components[:-1]), components[-1]
            if agg_type in ['pred', 'raw']:
               continue
            color = eye_colors[curr_eye % len(eye_colors)]
            curr_eye += 2
        # scores['hummingbird annotations response'] = [0.7895537653119531, 0.92024460851144, 0.9202031027601265, 0.9301341814931806]
        #print(method, scores[method])
        if method_name not in METHODS_TO_SHOW:
            continue
        n_shots = [0, 1, 2, 3]
        #ax.plot(n_shots, scores[method], color=color_map[method_name], linestyle=line_map[agg_type], label=method_name)
        #ax.fill_between(n_shots, [s - e for s, e in zip(scores[method], errs[method])], [s + e for s, e in zip(scores[method], errs[method])], color=color_map[method_name], alpha=0.5)
        ax.bar(np.asarray(n_shots) - 0.1 * group_num, scores[method], width=0.1, color=color_map[method_name],
               label=method_name)
        ax.errorbar(np.asarray(n_shots) - 0.1*group_num, scores[method], yerr=errs[method], color='black', capsize=3, linestyle='none')
        group_num += 1
        print(method_name)
        print(scores[method])
        ax.set_xlabel('N Shots')
        ax.set_ylabel('Average Confidence Score for Correct Class')

        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels([0, 1, 2, 4])
        ax.set_ylim([80, 100])
        ax.legend()

def plot_agg(score_results_sub, score_results_all, ax):
    n_shots = [0, 1, 2, 4]
    #for score_results_list, linestyle in [(score_results_sub, 'dashed'), score_results_all, 'solid']:
    scores_sub = {method: [] for method in score_results_sub[0].keys()}
    scores_all = {method: [] for method in scores_sub}
    errs_sub = {method: [] for method in score_results_sub[0].keys()}
    errs_all = {method: [] for method in errs_sub}
    for score_sub, score_all in zip(score_results_sub, score_results_all):
        for method, num in score_sub.items():
            totals = []
            for i in range(len(num[0])):
                sc = [1 if n[i] else 0 for n in num]
                totals.append(sum(sc)/len(sc))
            errs_sub[method].append(2*np.nanstd(totals) * 100)
            l = len((num))
            num = [1 for n in num if n.count(True) > 3]
            scores_sub[method].append(sum(num)/l * 100)

        for method, num in score_all.items():
            totals = []
            for i in range(len(num[0])):
                sc = [1 if n[i] else 0 for n in num]
                totals.append(sum(sc)/len(sc))
            errs_all[method].append(2*np.nanstd(totals) * 100)
            l = len((num))
            num = [1 for n in num if n.count(True) > 3]
            scores_all[method].append(sum(num)/l * 100)

    group_num = 0  # for bar plotting

    for method in scores_sub:
        if method in ['hummingbird annotations response', 'captum scores response', 'surprisal response', 'hybrid score', 'baseline']:
            components = method.split(' ')
            method_name = ' '.join(components[:-1])
            agg_type = 'non-eye-tracking'
            # if method.startswith('hummingbird'):
            #    score, op_score = 0.8280430306835632, 0.8955129357063171
            if method in ['hybrid score', 'baseline']:
                method_name = method.title()
                agg_type = 'hybrid'
        else:
            components = method.split(' ')[:-1]  # last component is just result
            method_name, agg_type = ' '.join(components[:-1]), components[-1]
            if agg_type in ['raw', 'pred']:
                continue
         #print(method, scores[method])
        if method_name in ['captum scores', 'Hybrid Score', 'surprisal', 'hummingbird annotations', 'Baseline']:
            continue
        n_shots = [0, 2, 4, 6]
        #ax.plot(n_shots, scores[method], color=color_map[method_name], linestyle=line_map[agg_type], label=method_name)
        ax.bar(np.asarray(n_shots) - 0.1 * group_num, scores_all[method], width=0.1, color=color_map[method_name],
               label=method_name, alpha=0.5, edgecolor=color_map[method_name])
        ax.set_xlabel('N Shots')
        ax.set_ylabel('Accuracy (%)')
        ax.set_xticks([-1.25, 0.75, 2.75, 4.75])
        ax.set_xticklabels([0, 1, 2, 4])
        #ax.set_yticks([375, 380, 385, 390])
        ax.set_ylim([70, 100])
        #ax.fill_between(n_shots, [s - e for s, e in zip(scores[method], errs[method])],
        #                [s + e for s, e in zip(scores[method], errs[method])], color=color_map[method_name], alpha=0.5)
        #
        ax.errorbar(np.asarray(n_shots) - 0.1*group_num, scores_all[method], yerr=errs_all[method], color='black', capsize=3, linestyle='none')
        group_num += 1

        ax.bar(np.asarray(n_shots) - 0.1 * group_num, scores_sub[method], width=0.1, color=color_map[method_name],
               hatch='/', edgecolor='black', alpha=0.3)
        ax.errorbar(np.asarray(n_shots) - 0.1 * group_num, scores_sub[method], errs_sub[method], color='black',
                    capsize=3, linestyle='none')
        group_num += 2
        handles, labels = ax.get_legend_handles_labels()
        fx = lambda h: mpatches.Patch(facecolor='white', edgecolor='black', hatch=h)
        hatch_handles = [fx(''), fx('/')]

        hatch_labels = ['All Data', 'Incongruent - Congruent']

        legend = ax.legend(handles[::-1] + hatch_handles, labels[::-1] + hatch_labels, ncol=2, framealpha=1,
                           loc='lower left', fontsize=17)
        ax.add_artist(legend)


def plot_agg_score(score_results_sub, score_results_all, ax):
    n_shots = [0, 1, 2, 4]
    #for score_results_list, linestyle in [(score_results_sub, 'dashed'), score_results_all, 'solid']:
    scores_sub = {method: [] for method in score_results_sub[0].keys()}
    scores_all = {method: [] for method in scores_sub}
    errs_sub = {method: [] for method in score_results_sub[0].keys()}
    errs_all = {method: [] for method in errs_sub}
    for score_sub, score_all in zip(score_results_sub, score_results_all):
        for method, num in score_sub.items():
            totals = []
            for i in range(len(num[0])):
                sc = [n[i] for n in num]
                totals.append(sum(sc)/len(sc))
            errs_sub[method].append(2*np.nanstd(totals))
            num = flatten(num)
            l = len((num))
            scores_sub[method].append(sum(num)/l)

        for method, num in score_all.items():
            totals = []
            for i in range(len(num[0])):
                sc = [n[i] for n in num]
                totals.append(sum(sc)/len(sc))
            errs_all[method].append(2*np.nanstd(totals))
            num = flatten(num)
            l = len((num))
            scores_all[method].append(sum(num)/l)

    group_num = 0  # for bar plotting

    for method in scores_sub:
        if method in ['hummingbird annotations response', 'captum scores response', 'surprisal response', 'hybrid score', 'baseline']:
            components = method.split(' ')
            method_name = ' '.join(components[:-1])
            agg_type = 'non-eye-tracking'
            # if method.startswith('hummingbird'):
            #    score, op_score = 0.8280430306835632, 0.8955129357063171
            if method in ['hybrid score', 'baseline']:
                method_name = method.title()
                agg_type = 'hybrid'
        else:
            components = method.split(' ')[:-1]  # last component is just result
            method_name, agg_type = ' '.join(components[:-1]), components[-1]
            if agg_type in ['raw', 'pred']:
                continue
         #print(method, scores[method])
        if method_name in ['captum scores', 'hybrid score', 'surprisal', 'hummingbird annotations']:
            continue
        n_shots = [0, 2, 4, 6]
        #ax.plot(n_shots, scores[method], color=color_map[method_name], linestyle=line_map[agg_type], label=method_name)
        ax.bar(np.asarray(n_shots) - 0.1 * group_num, scores_all[method], width=0.1, color=color_map[method_name],
               label=method_name, alpha=0.7)
        ax.set_xlabel('N Shots')
        ax.set_ylabel('Average Confidence Score')
        #ax.set_yticks([375, 380, 385, 390])
        #ax.set_ylim([370, 395])
        #ax.fill_between(n_shots, [s - e for s, e in zip(scores[method], errs[method])],
        #                [s + e for s, e in zip(scores[method], errs[method])], color=color_map[method_name], alpha=0.5)
        #
        ax.errorbar(np.asarray(n_shots) - 0.1*group_num, scores_all[method], yerr=errs_all[method], color=color_map[method_name], capsize=3, linestyle='none')
        group_num += 1

        if method_name in ['hummingbird annotations']:
            continue
        ax.bar(np.asarray(n_shots) - 0.1 * group_num, scores_sub[method], width=0.1, color=color_map[method_name],
               label=method_name + ' (inc - con)', hatch='/', edgecolor='black', alpha=0.7)
        ax.errorbar(np.asarray(n_shots) - 0.1 * group_num, scores_sub[method], errs_sub[method], color=color_map[method_name],
                    capsize=3, linestyle='none')
        group_num += 2
        ax.set_xticks([0, 2, 4, 6])
        ax.set_xticklabels([0, 1, 2, 4])
        ax.legend()

def plot_eye(opposite_results, score_results, ax):
    eye_colors = ['midnightblue', 'darkslateblue', 'blue', 'slateblue', 'royalblue', 'cornflowerblue', 'dodgerblue', 'lightskyblue',
                  'darkturquoise', 'cyan', 'aquamarine', 'cadetblue', 'steelblue', 'mediumslateblue']
    curr_eye = 0
    non_eye_colors = ['orange', 'gold', 'brown', 'darkgoldenrod', 'black']
    curr_non_eye = 0
    x = []
    y = []
    errors = []
    x_ = 0
    for method, num in score_results.items():
        num = [n for n in num if n > 0.5]
        score = sum(num)/len(num)
        print(len(num))
        ci = stats.norm.interval(alpha=0.95, loc=stats.gmean(num), scale=np.std(num))
        op_num = opposite_results[method]
        op_score = median(num)
        #op_score = sum(op_num)/len(op_num)
        if method == 'baseline':
            method_name = method.title()
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

def write_file(score_results_list, acc):
    n_shots = [0, 1, 2, 4]
    scores = {method: [] for method in score_results_list[0].keys()}
    errs = {method: [] for method in score_results_list[0].keys()}
    for score_results in score_results_list:
        for method, num in score_results.items():
            totals = []
            for i in range(len(num[0])):
                sc = [1 if n[i] else 0 for n in num]
                totals.append(sum(sc)/len(sc))
            errs[method].append(2*np.nanstd(totals) * 100)
            l = len((num))
            num = [1 for n in num if n.count(True) > 3 ]
            scores[method].append(sum(num)/l * 100)
    with open(f'fewshot_results_{acc}.csv', 'w') as out:
        writer = csv.writer(out)
        methods = list(scores.keys())
        for method, results in scores.items():
            row = results
            for i in range(len(row)):
                row[i] = f"{row[i]: .2f} ({errs[method][i]: .2f})"
            writer.writerow([method] + row)



if __name__ == '__main__':
    accumulators = [Accumulator.SUBTRACTIVE, Accumulator.ALL]
    criterions = [Criterion.AVG]
    x_i = 0
    y_i = 0
    acc_all = []
    acc_sub = []
    for a in accumulators:
        #zero_shots = ZERO_SHOT_SUBTRACTIVE if a == Accumulator.SUBTRACTIVE else ZERO_SHOT_ALL
        #ax = axes[x_i]
        #x_i += 1
        #print(x_i, y_i)
        f, axes = plt.subplots(figsize=(10, 7))
        ax = axes
        foundfile = False
        for c in criterions:
            all_score_results = []
            all_accuracy_results = []
            for n in shots:
                filename = 'openai_{}shot_{}_{}.csv'.format(n, c, a)
                if not os.path.exists(filename):
                    continue
                foundfile = True
                accuracy, score_results = score_file(filename)

                filename = 'openai_{}shot_hybrid_{}_{}'.format(n, c, a)
                hybrid_accuracy, hybrid_results = score_file(filename)

                filename = 'openai_{}shot_baseline_{}_{}'.format(n, c, a)
                baseline_acc, baseline_res = score_file(filename)
                print(baseline_res.keys())

                score_results['hybrid score'] = hybrid_results['combined score']
                score_results['baseline'] = baseline_res['baseline response']
                accuracy['hybrid score'] = hybrid_accuracy['combined score']
                accuracy['baseline'] = baseline_acc['baseline response']
                all_score_results.append(score_results)
                all_accuracy_results.append(accuracy)
                if a == Accumulator.ALL:
                    acc_all = all_accuracy_results
                    sc_all = all_score_results
                else:
                    acc_sub = all_accuracy_results
                    sc_sub = all_score_results
            #bar_plot(all_score_results, ax)
            plot(all_score_results, all_accuracy_results, ax)

            #ax.set_title(f'Eye Data: {"All" if a == Accumulator.ALL else "Incongruent - Congruent "}', fontsize='small', loc='left')
            ax.set_xticklabels([0, 1, 2, 4])
            plt.tight_layout()
            plt.show()
            f, ax = plt.subplots(figsize=(10, 7))
            #ax.set_title(f'Eye Data: {"All" if a == Accumulator.ALL else "Incongruent - Congruent "}', fontsize='small',
            #             loc='left')
            ax.set_xticklabels([0, 1, 2, 4])
            bar_plot(all_accuracy_results, ax)
            plt.tight_layout()
            plt.show()

            #ax.set_ylim([0.90, 1.0])
            #ax.set_xlim([.77, .82])
            #ax.set_ylim([.72, .79])


    #handles, labels = axes[0][0].get_legend_handles_labels()
    #fx = lambda m, c: plt.plot([], [], marker=m, color=c, ls="none")[0]
    #colors = color_map.values()
    #markers = marker_map.values()
    #handles1 = [fx("s", color) for color in colors]
    #handles2 = [fx(marker, "k") for marker in markers]

    #labels1 = list(color_map.keys())
    #labels2 = list(marker_map.keys())

    #legend1 = plt.legend(handles1, labels1, loc='lower left')
    #legend2 = plt.legend(handles2, labels2, loc='lower right')
    #plt.gca().add_artist(legend1)
    #plt.gca().add_artist(legend2)
    # plt.xticks(rotation=30, horizontalalignment='right')
    #plt.autoscale(True, axis='both', tight=True)
    f, ax = plt.subplots(figsize=(10, 7))
    write_file(acc_all, 'all')
    write_file(acc_sub, 'sub')
    plot_agg(acc_all, acc_sub, ax)
    #plt.legend()
    plt.tight_layout()
    plt.show()
