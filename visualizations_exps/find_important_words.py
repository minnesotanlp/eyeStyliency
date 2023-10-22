import csv
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import pandas as pd
from transformers import pipeline, RobertaTokenizer, AutoModelForSequenceClassification
import numpy as np
import json
from utils.utils import *
from ia_processing.ia_processing_helpers import *
import os

DATA_FILENAME = '../data/eyelink_data_normalized.csv'
STIMULI_FILENAME = 'ppl.csv'
df = pd.read_csv(DATA_FILENAME)
NOSTOPS = True
SAVE_METRICS_NAME = '.tmpmetrics'

# criterion can be AVG, TOP_10, TOP_20
criterion = Criterion.TOP_33

# accumulator can be SUBTRACTIVE or ALL
accum = Accumulator.SUBTRACTIVE

if os.path.exists(SAVE_METRICS_NAME):
    with open(SAVE_METRICS_NAME) as f:
        m = json.load(f)

out = 'importantwords/words_{}_{}.csv'.format(criterion, accum)
if os.path.exists(out):
    print("File already exists! Exiting.")
    exit()

with open(STIMULI_FILENAME) as f, open(out, 'w') as out:
    reader = csv.reader(f)
    writer = csv.writer(out)
    header_row = ['stim_id', 'text', 'style']
    lines = list(reader)
    header = lines[0]
    id_idx, label_idx, type_idx, text_idx, hb_idx, cp_idx, outpt_idx, conf_idx, ppl_idx = get_indices(header)
    stims = lines[1:]  # first line is header
    for metric in m:
        header_row.append(metric)
    writer.writerow(header_row)
    style_options = [["Negative", "Positive"], ["Polite", "Impolite"]]

    for stim in stims:
        stim_id = int(stim[1])
        style = stim[label_idx].split(' ')[0]
        options = style_options[0] if style in style_options[0] else style_options[1]
        text = stim[text_idx]
        cleaned_text = [clean_word(w) for w in stim[3].split(' ')]
        maxIA = get_max_IA(stim_id)
        out_line = [stim_id, text, style]

        for metric in m:
            metric_parts = metric.split(' ')
            subtype = metric_parts[-1]
            meas = ' '.join(metric_parts[:-1])
            if meas in measures:  # it's an eyetracking metric
                # get data according to criterion
                if accum is Accumulator.SUBTRACTIVE:
                    data = get_sub_scores(stim_id, meas, subtype, text)
                elif accum is Accumulator.ALL:
                    data = get_score(stim_id, meas, subtype, text)
                metric_name = meas + ' ' + subtype
            else:
                if metric == 'hummingbird annotations':
                    data = [abs(n) for n in tolist(stim[hb_idx])]
                if metric == 'captum scores':
                    data = tolist(stim[cp_idx])
                if metric == 'surprisal':
                    data = tolist(stim[ppl_idx])
                if NOSTOPS:
                    data = merge_baseline(stim_id, text, data)
                metric_name = metric

            if len(data) != len(ia_indices(stim_id, text)):
                out_line.append('NA')
                print('mismatched ias found:')
                print('stim: {},metric:{} data:{}, indices{}'.format(stim_id, metric, len(data), len(ia_indices(stim_id, text))))
                continue
            if criterion is Criterion.AVG:
                threshold = get_threshold(m[metric_name], criterion)
            else:
                threshold = get_threshold([data], criterion)
            data = np.asarray(data)
            data = np.nan_to_num(data)
            important_words = []
            word_scores = []
            for i, ia_id in enumerate(ia_indices(stim_id, text)):
                stim_df = df.loc[(df['stim_id'] == stim_id)]
                ia_id = ia_id[0]
                ia_labels = stim_df.loc[stim_df['IA_ID'] == ia_id]['IA_LABEL']
                label = pd.unique(ia_labels)[0]
                word_scores.append((' '.join(label.split('_')), data[i]))
                if data[i] > threshold:
                    important_words.append(' '.join(label.split('_')))
            out_line.append(important_words)
        writer.writerow(out_line)

