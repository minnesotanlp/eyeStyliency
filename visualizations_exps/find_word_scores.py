from ia_processing.ia_processing_helpers import *
import os

DATA_FILENAME = '../data/eyelink_data_normalized.csv'
STIMULI_FILENAME = 'ppl.csv'
SAVE_METRICS_NAME= '.tmpmetrics'
df = pd.read_csv(DATA_FILENAME)
NOSTOPS = True

# criterion can be AVG, TOP_10, TOP_20
criterion = Criterion.EXP

# accumulator can be SUBTRACTIVE or ALL
accum = Accumulator.SUBTRACTIVE

if os.path.exists(SAVE_METRICS_NAME):
    with open(SAVE_METRICS_NAME) as f:
        m = json.load(f)

out = 'wordscores/wordscores_{}_{}.csv'.format(criterion, accum)
if os.path.exists(out):
    print("File already exists! Exiting.")
    #exit()

with open(STIMULI_FILENAME) as f, open(out, 'w') as out:
    reader = csv.reader(f)
    results = {}
    lines = list(reader)
    header = lines[0]
    id_idx, label_idx, type_idx, text_idx, hb_idx, cp_idx, outpt_idx, conf_idx, ppl_idx = get_indices(header)
    stims = lines[1:]  # first line is header
    for metric in m:
        results[metric] = []

    for stim in stims:
        stim_id = int(stim[1])
        print(stim_id)
        style = stim[label_idx].split(' ')[0]
        text = stim[text_idx]
        cleaned_text = [clean_word(w) for w in stim[3].split(' ')]
        out_line = [stim_id, text, style]

        for i, metric in enumerate(m):
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
                    stim_id == 37 and print(data)
                if metric == 'captum scores':
                    data = tolist(stim[cp_idx])
                if metric == 'surprisal':
                    data = tolist(stim[ppl_idx])
                if NOSTOPS:
                    data = merge_baseline(stim_id, text, data)
                metric_name = metric

            if len(data) != len(ia_indices(stim_id, text)):
                print('stim: {},metric:{} data:{}, indices{}'.format(stim_id, metric, len(data), len(ia_indices(stim_id, text))))
                continue
            if criterion in [Criterion.AVG, Criterion.EXP]:
                met = np.asarray(m[metric_name])
                mean = np.nanmean(met)
                std = np.nanstd(met)
                met = np.where(met < (mean + 3 * std), met, np.nan)

                threshold = np.nanmean(met)
            else:
                threshold = get_item_threshold(data, criterion)
            if metric == 'hummingbird annotations':
                threshold = 0
            data = np.asarray(data)
            data = np.nan_to_num(data)
            important_words = []
            ia_idx = list(ia_indices(stim_id, text))
            for i, ia_id in enumerate(ia_idx):
                stim_df = df.loc[(df['stim_id'] == stim_id)]
                ia_id = ia_id[0]
                ia_labels = stim_df.loc[stim_df['IA_ID'] == ia_id]['IA_LABEL']
                label = pd.unique(ia_labels)[0]
                if data[i] > threshold:
                    important_words.append((' '.join(label.split('_')), data[i], style))
                else:
                    important_words.append((' '.join(label.split('_')), 0, style))
            results[metric] += [important_words]
    print(results.keys())
    print(len(results))
    out.write(json.dumps(results))
