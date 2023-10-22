import csv
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import pandas as pd
from transformers import pipeline, RobertaTokenizer, AutoModelForSequenceClassification
import numpy as np
import openai
import json
import random
from utils.utils import *
from ia_processing.ia_processing_helpers import *
import os

assert os.environ.get('OPENAI_API_KEY') is not None, 'Environment variable OPENAI_API_KEY not found.'
openai.api_key = os.environ.get('OPENAI_API_KEY')
DATA_FILENAME = '../../data/eyelink_data_normalized.csv'
STIMULI_FILENAME = '../ppl.csv'
df = pd.read_csv(DATA_FILENAME)
NOSTOPS = True
all_stims = pd.read_csv('../../stimuli/EyetrackingStimuli.csv', index_col=False)


def get_important_words(stim_id, meas, subtype, text, accum=Accumulator.SUBTRACTIVE, crit=Criterion.AVG):
    if accum is Accumulator.SUBTRACTIVE:
        data = get_sub_scores(stim_id, meas, subtype, text)
    elif accum is Accumulator.ALL:
        data = get_score(stim_id, meas, subtype, text)
    if criterion is Criterion.AVG:
        threshold = np.nanmean(m[meas + ' ' + subtype])
    else:
        threshold = get_item_threshold(data, criterion)
    data = np.asarray(data)
    data = np.nan_to_num(data)
    important_words = []
    for i, ia_id in enumerate(ia_indices(stim_id, text)):
        stim_df = df.loc[(df['stim_id'] == stim_id)]
        ia_id = ia_id[0]
        ia_labels = stim_df.loc[stim_df['IA_ID'] == ia_id]['IA_LABEL']
        label = pd.unique(ia_labels)[0]
        if data[i] > threshold:
            important_words.append((' '.join(label.split('_')), data[i], style))
    return important_words


def get_example_stims(options, n, stim_id):
    cleaned_options = []
    for opt in options:
        if opt in ["Negative", "Positive"]:
            cleaned_options.append(opt + " Sentiment")
        else:
            cleaned_options.append(opt)
    examples = all_stims.loc[all_stims['Label'].isin(cleaned_options)]
    ids = examples['id'].tolist()
    ids.remove(stim_id)
    idxs = []
    while len(idxs) != n:
        random_idx = random.randint(0, len(ids) - 1)
        random_stim = ids[random_idx]
        if random_stim not in idxs:
            idxs.append(random_stim)
    return idxs


def get_text(stim_id):
    return all_stims.loc[all_stims['id'] == stim_id]['Text'].tolist()[0]


def get_style(stim_id):
    return all_stims.loc[all_stims['id'] == stim_id]['Label'].tolist()[0].split(' ')[0]


def get_words(stim, hb_idx, cp_idx, ppl_idx, metric, stim_id, text):
    if metric == 'hummingbird annotations':
        data = [abs(n) for n in tolist(stim[hb_idx])]
    if metric == 'captum scores':
        data = tolist(stim[cp_idx])
    if metric == 'surprisal':
        data = tolist(stim[ppl_idx])
    if NOSTOPS:
        data = merge_baseline(stim_id, text, data)
    data = np.asarray(data)
    data = np.nan_to_num(data)
    important_words = []
    for i, ia_id in enumerate(ia_indices(stim_id, text)):
        stim_df = df.loc[(df['stim_id'] == stim_id)]
        ia_id = ia_id[0]
        ia_labels = stim_df.loc[stim_df['IA_ID'] == ia_id]['IA_LABEL']
        label = pd.unique(ia_labels)[0]
        if data[i] > threshold:
            important_words.append(' '.join(label.split('_')))
    return important_words


def create_demo(text, important_words, options, correct_option=None):
    option1, option2 = options
    if correct_option:
        correct_option = ' ' + correct_option
    if not important_words:
        return f"Decide whether the text style is {option1} or {option2}.\nText: {text}\n" \
               f"{option1} or {option2}:{correct_option}"
    important_words = ', '.join(important_words)
    return f"Decide whether the text style is {option1} or {option2}.\nText: {text}\nImportant words:" \
           f" {important_words}\n{option1} or {option2}:{correct_option}"


def create_prompt(text, important_words, options):
    if not important_words:
        return f"Decide whether the text style is {options[0]} or {options[1]}.\nText: {text}\n" \
               f"{options[0]} or {options[1]}:"
    return f"Decide whether the text style is {options[0]} or {options[1]}.\nText: {text}\nImportant words:" \
           f" {', '.join(important_words)}\n{options[0]} or {options[1]}:"


def get_response(prompt):
    if prompt in ALL_RESP:
        print('cached!')
        cached_dict = json.loads(ALL_RESP[prompt])
        if "original_prompt" not in cached_dict:
            cached_dict["original_prompt"] = prompt
        return json.dumps(cached_dict)
    print('not cached')
    try:
        response = openai.Completion.create(
            model="text-davinci-002",
            prompt=prompt,
            temperature=0,
            max_tokens=60,
            top_p=1.0,
            logprobs=5,
        )
    except:  #timeout exception
        print('there was an exception, retrying')
        return get_response(prompt)
    response_dict = response.to_dict_recursive()
    response_dict['original_prompt'] = prompt
    response = json.dumps(response_dict)
    ALL_RESP[prompt] = response
    write_cache()
    return response


def write_cache():
    with open('resp_cache.txt', 'w') as cache:
        cache.write(json.dumps(ALL_RESP))


def get_baseline_response(text, options):
    prompt = "Decide whether the text style is " + options[0] + " or " + options[1] \
             + "\nText: " + text + "\n" + options[0] + " or " + \
             options[1] + ":"
    if prompt in ALL_RESP:
        return ALL_RESP[prompt]
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=prompt,
        temperature=0,
        max_tokens=60,
        top_p=1.0,
        logprobs=5,
        frequency_penalty=0.5,
        presence_penalty=0.0
    )
    response = json.dumps(response.to_dict_recursive())
    ALL_RESP[prompt] = response
    return response



# criterion can be AVG, TOP_10, TOP_20
# criterion = Criterion.TOP_33

# accumulator can be SUBTRACTIVE or ALL
# accum = Accumulator.ALL

N_SHOTS = 2  # how many examples to do
N_ROUNDS = 5

if os.path.exists(SAVE_METRICS_NAME):
    with open(SAVE_METRICS_NAME) as f:
        m = json.load(f)

with open('baselines.csv') as f:
    reader = csv.reader(f)
    baselines = list(reader)[1:]
    baselines = [l[-1] for l in baselines]

with open('resp_cache.txt') as f:
    ALL_RESP = json.load(f)

for i in [0, 1, 2, 4]:
    N_SHOTS = i
    criterion = Criterion.AVG
    random.seed(2)
    for accum in [Accumulator.ALL, Accumulator.SUBTRACTIVE]:
        out = 'openai_{}shot_baseline_{}_{}'.format(N_SHOTS, criterion, accum)
        if os.path.exists(out):
            print("File already exists! Continuing.")
            continue
        with open(STIMULI_FILENAME) as f, open(out, 'w') as out:
            reader = csv.reader(f)
            writer = csv.writer(out)
            header_row = ['stim_id', 'text', 'style', 'baseline score', 'baseline response']
            lines = list(reader)
            header = lines[0]
            id_idx, label_idx, type_idx, text_idx, hb_idx, cp_idx, outpt_idx, conf_idx, ppl_idx = get_indices(header)
            stims = lines[1:]  # first line is header
            writer.writerow(header_row)
            style_options = [["Negative", "Positive"], ["Polite", "Impolite"]]

            for stim in stims:
                stim_id = int(stim[1])
                print(stim_id)
                style = stim[label_idx].split(' ')[0]
                options = style_options[0] if style in style_options[0] else style_options[1]
                text = stim[text_idx]
                cleaned_text = [clean_word(w) for w in stim[3].split(' ')]
                out_line = [stim_id, text, style]

                demos = get_example_stims(options, N_SHOTS * N_ROUNDS, stim_id)

                hb = [abs(n) for n in tolist(stim[hb_idx])]
                hb = merge_baseline(stim_id, text, hb)
                if accum is Accumulator.SUBTRACTIVE:
                    data = get_sub_scores(stim_id, 'Dwell Time', 'z_score', text)
                elif accum is Accumulator.ALL:
                    data = get_score(stim_id, 'Dwell Time', 'z_score', text)

                if len(data) != len(ia_indices(stim_id, text)):
                    out_line.append('NA')  # na imp words
                    out_line.append('NA')  # na result
                    print('stim: {},metric:{} data:{}, indices{}'.format(stim_id, 'metric', len(data),
                                                                         len(ia_indices(stim_id, text))))
                    continue

                if len(data) != len(hb):
                    continue

                out_line.append([])

                idx = 0
                results = []
                for i in range(N_ROUNDS):
                    prompt = ''
                    for j in range(N_SHOTS):  # For each shot, add a demo
                        demo_id = demos[idx]
                        idx += 1
                        demo_text = get_text(demo_id)
                        demo_style = get_style(demo_id)
                        prompt += create_demo(demo_text, None, options, demo_style)
                        prompt += '\n\n'
                    prompt += create_prompt(text, None, options)
                    result = get_response(prompt)
                    results.append(result)
                out_line.append(results)
                writer.writerow(out_line)

for i in [0, 1, 2, 4]:
    N_SHOTS = i
    criterion = Criterion.AVG
    random.seed(2)
    for accum in [Accumulator.ALL, Accumulator.SUBTRACTIVE]:
        out = 'openai_responses/openai_{}shot_hybrid_{}_{}'.format(N_SHOTS, criterion, accum)
        if os.path.exists(out):
            print("File already exists! Continuing.")
            continue
        with open(STIMULI_FILENAME) as f, open(out, 'w') as out:
            reader = csv.reader(f)
            writer = csv.writer(out)
            header_row = ['stim_id', 'text', 'style', 'important words', 'combined score']
            lines = list(reader)
            header = lines[0]
            id_idx, label_idx, type_idx, text_idx, hb_idx, cp_idx, outpt_idx, conf_idx, ppl_idx = get_indices(header)
            stims = lines[1:]  # first line is header
            writer.writerow(header_row)
            style_options = [["Negative", "Positive"], ["Polite", "Impolite"]]

            for stim in stims:
                stim_id = int(stim[1])
                print(stim_id)
                style = stim[label_idx].split(' ')[0]
                options = style_options[0] if style in style_options[0] else style_options[1]
                text = stim[text_idx]
                cleaned_text = [clean_word(w) for w in stim[3].split(' ')]
                out_line = [stim_id, text, style]

                demos = get_example_stims(options, N_SHOTS * N_ROUNDS, stim_id)

                hb = [abs(n) for n in tolist(stim[hb_idx])]
                hb = merge_baseline(stim_id, text, hb)
                if accum is Accumulator.SUBTRACTIVE:
                    data = get_sub_scores(stim_id, 'Dwell Time', 'z_score', text)
                elif accum is Accumulator.ALL:
                    data = get_score(stim_id, 'Dwell Time', 'z_score', text)

                if len(data) != len(ia_indices(stim_id, text)):
                    out_line.append('NA')  # na imp words
                    out_line.append('NA')  # na result
                    print('stim: {},metric:{} data:{}, indices{}'.format(stim_id, 'metric', len(data),
                                                                         len(ia_indices(stim_id, text))))
                    continue
                hb_threshold = np.nanmean(m['hummingbird annotations'])

                met = np.asarray(m['Dwell Time z_score'])
                mean = np.nanmean(met)
                std = np.nanstd(met)
                met = np.where(met < mean + (3 * std), met, np.nan)
                threshold = np.nanmean(met)

                important_words = []
                if len(data) != len(hb):
                    continue

                for i, ia_id in enumerate(ia_indices(stim_id, text)):
                    stim_df = df.loc[(df['stim_id'] == stim_id)]
                    ia_id = ia_id[0]
                    ia_labels = stim_df.loc[stim_df['IA_ID'] == ia_id]['IA_LABEL']
                    label = pd.unique(ia_labels)[0]
                    if data[i] > threshold or hb[i] > hb_threshold:
                        important_words.append(' '.join(label.split('_')))

                out_line.append(important_words)
                if len(important_words) > 0:
                    idx = 0
                    results = []
                    for i in range(N_ROUNDS):
                        prompt = ''
                        for j in range(N_SHOTS):  # For each shot, add a demo
                            demo_id = demos[idx]
                            idx += 1
                            demo_text = get_text(demo_id)
                            demo_style = get_style(demo_id)
                            ws = get_important_words(demo_id, 'Dwell Time', 'z_score', demo_text, accum, criterion)
                            ws2 = get_words(stim, hb_idx, cp_idx, ppl_idx, 'hummingbird annotations', demo_id, demo_text)
                            demo_words = list(set(ws) & set(ws2))
                            prompt += create_demo(demo_text, demo_words, options, demo_style)
                            prompt += '\n\n'
                        prompt += create_prompt(text, important_words, options)
                        result = get_response(prompt)
                        results.append(result)
                out_line.append(results)
                writer.writerow(out_line)


for i in [0, 1, 2, 4]:
    random.seed(2)
    N_SHOTS = i
    for criterion in [Criterion.AVG]:
        for accum in [Accumulator.ALL, Accumulator.SUBTRACTIVE]:
            out = 'openai_responses/openai_{}shot_{}_{}.csv'.format(N_SHOTS, criterion, accum)
            if os.path.exists(out):
                print("File already exists! Continuing.")
                continue

            with open(STIMULI_FILENAME) as f, open(out, 'w') as out:
                reader = csv.reader(f)
                writer = csv.writer(out)
                header_row = ['stim_id', 'text', 'style', 'baseline']
                lines = list(reader)
                header = lines[0]
                id_idx, label_idx, type_idx, text_idx, hb_idx, cp_idx, outpt_idx, conf_idx, ppl_idx = get_indices(header)
                stims = lines[1:]  # first line is header
                for metric in m:
                    header_row.append(metric + ' important words')
                    header_row.append(metric + ' response')
                writer.writerow(header_row)
                style_options = [["Negative", "Positive"], ["Polite", "Impolite"]]

                all_hb = np.asarray([abs(n) for n in flatten([tolist(line[hb_idx]) for line in stims])])
                all_cptm = np.asarray(flatten([tolist(line[cp_idx]) for line in stims]))
                all_ppl = np.asarray(flatten([tolist(line[ppl_idx]) for line in stims]))

                for stim in stims:
                    stim_id = int(stim[1])
                    print(stim_id)
                    style = stim[label_idx].split(' ')[0]
                    options = style_options[0] if style in style_options[0] else style_options[1]
                    text = stim[text_idx]
                    cleaned_text = [clean_word(w) for w in stim[3].split(' ')]
                    out_line = [stim_id, text, style]

                    # get baseline
                    #baseline = get_baseline_response(text, options)
                    baseline = ''
                    out_line.append(baseline)

                    demos = get_example_stims(options, N_SHOTS * N_ROUNDS, stim_id)

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
                            out_line.append('NA')  # na imp words
                            out_line.append('NA')  # na result
                            print('stim: {},metric:{} data:{}, indices{}'.format(stim_id, metric, len(data), len(ia_indices(stim_id, text))))
                            continue
                        if criterion is Criterion.AVG:
                            if metric in ['hummingbird annotations', 'captum scores', 'surprisal']:
                                threshold = np.nanmean(m[metric_name])
                            else:
                                if accum == Accumulator.SUBTRACTIVE:
                                    alld = get_all_sub_scores(meas, subtype)
                                else:
                                    alld = m[metric]
                                met = np.asarray(alld)
                                mean = np.nanmean(met)
                                std = np.nanstd(met)
                                met = np.where(met < mean + (3 * std), met, np.nan)
                                threshold = np.nanmean(met)
                        else:
                            threshold = get_item_threshold(data, criterion)
                        data = np.asarray(data)
                        data = np.nan_to_num(data)
                        important_words = []
                        for i, ia_id in enumerate(ia_indices(stim_id, text)):
                            stim_df = df.loc[(df['stim_id'] == stim_id)]
                            ia_id = ia_id[0]
                            ia_labels = stim_df.loc[stim_df['IA_ID'] == ia_id]['IA_LABEL']
                            label = pd.unique(ia_labels)[0]
                            if data[i] > threshold:
                                important_words.append(' '.join(label.split('_')))

                        result = baseline
                        out_line.append(important_words)
                        if len(important_words) > 0:
                            idx = 0
                            results = []
                            for i in range(N_ROUNDS):
                                prompt = ''
                                for j in range(N_SHOTS):  # For each shot, add a demo
                                    demo_id = demos[idx]
                                    idx += 1
                                    demo_text = get_text(demo_id)
                                    demo_style = get_style(demo_id)
                                    if meas in measures:
                                        ws = get_important_words(demo_id, meas, subtype, demo_text, accum, criterion)
                                        ws = [w[0] for w in ws]
                                    else:
                                        ws = get_words(stim, hb_idx, cp_idx, ppl_idx, metric, demo_id, demo_text)
                                    prompt += create_demo(demo_text, ws, options, demo_style)
                                    prompt += '\n\n'
                                prompt += create_prompt(text, important_words, options)
                                result = get_response(prompt)
                                results.append(result)
                        out_line.append(results)
                    writer.writerow(out_line)

