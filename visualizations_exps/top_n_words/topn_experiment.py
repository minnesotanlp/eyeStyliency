import csv
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import pandas as pd

DATA_FILENAME = '../eyelink_data_normalized.csv'
STIMULI_FILENAME = 'ppl.csv'
df = pd.read_csv(DATA_FILENAME)

eye_measures = {'dwell time': ['IA_DWELL_TIME', 'subDT'],
                'first fixation duration': ['IA_FIRST_FIXATION_DURATION', 'subFFD'],
                'first run dwell time': ['IA_FIRST_RUN_DWELL_TIME', 'subFRD'],
                'last run dwell time': ['IA_LAST_RUN_DWELL_TIME', 'subLRD'],
                'regressions out': ['IA_REGRESSION_OUT_FULL_COUNT', 'subRO']}


def tolist(item):
    item = item.strip()
    if item == 'NA':
        return []
    result = [i if i != 'nan' else 0 for i in item.split(' ')]
    return [float(i) for i in result if i]

def get_max_IA(df, stim_id):
    ias = df.loc[(df['stim_id'] == stim_id)]['IA_ID'].tolist()
    return sorted(ias)[-1]


def get_statistic(df, stim_id, text, measure):
    """gets list of ia-ordered averaged metric for subtractive, congruent, and incongruent case"""
    max_IA = get_max_IA(df, stim_id)
    sub = [0 for _ in range(max_IA)]
    con = [0 for _ in range(max_IA)]
    incon = [0 for _ in range(max_IA)]

    measure_name, sub_name = measure

    stim_df = df.loc[(df['stim_id'] == stim_id)]

    for ia_id in range(1, max_IA + 1):
        idx = ia_id - 1
        ia_df = stim_df.loc[stim_df['IA_ID'] == ia_id]
        cons = ia_df.loc[ia_df['congruent'] == True]
        con[idx] = cons[measure_name].mean(skipna=True)
        incons = ia_df.loc[ia_df['congruent'] == False]
        incon[idx] = incons[measure_name].mean(skipna=True)
        sub[idx] = incons[measure_name].mean(skipna=True) - cons[measure_name].mean(skipna=True)

    return sub, con, incon


def clean_word(word):
    return word.lower().strip(' ,"\'.`\n?!@#¬†;():*.')


def is_constant(measure):
    prev = measure[0]
    for m in measure:
        if m != prev:
            return False
    return True


def flatten(l):
    return [item for sublist in l for item in sublist]


def clean_topn(dct):
    """takes dict of words:all_scores, returns sorted list of [word, avg score]
    and removes any words with fewer than three scores"""
    result = []
    for measure in dct:
        to_pop = []
        for word, lst in measure.items():
            if WORD_COUNTS[word] < 3 or word in stopwords.words('english'):  # do not consider words that don't appear at least 2 times
                to_pop.append(word)
            else:
                # lst = flatten(lst)
                lst = [abs(elt) for elt in lst]
                measure[word] = sum(lst)
        for word in to_pop:
            measure.pop(word)
        result.append(sorted(measure.items(), key=lambda item: item[1]))
    return result


def find_move_indices(words):
    idx_to_move = []
    for ia_id in range(len(words)):
        if words[ia_id] in stopwords.words('english'):
            j = ia_id
            left_idx, right_idx = None, None
            while j >= 0:
                j -= 1
                if words[j] not in stopwords.words('english'):
                    left_idx = j
                    break
            j = ia_id
            while j < len(words):
                if words[j] not in stopwords.words('english'):
                    right_idx = j
                j += 1
            if left_idx and right_idx:
                idx = left_idx if ia_id - left_idx < ia_id - right_idx else right_idx  # right wins in case of tie
            else:
                idx = left_idx if left_idx else right_idx
            idx_to_move.append((ia_id, idx))
    return idx_to_move


def move_indices(datas, idx_to_move):
    for data in datas:
        for frm, to in idx_to_move:
            data[to] += data[frm]
        indices_to_rm = [frm for frm, to in idx_to_move]
        for idx in sorted(indices_to_rm, reverse=True):
            del data[idx]

WORD_COUNTS = {}

# get top n words for each style
with open(STIMULI_FILENAME) as f, open('topn.csv', 'w') as out:
    reader = csv.reader(f)
    writer = csv.writer(out)
    stims = list(reader)[1:]  # first line is header


    eye_measure_names = [name for name in eye_measures.keys()]
    metrics = eye_measure_names + ['cptm', 'ppl', 'hb']

    top_neg = [{} for _ in metrics]
    top_pos = [{} for _ in metrics]
    top_pol = [{} for _ in metrics]
    top_imp = [{} for _ in metrics]
    top_lists = [top_neg, top_pos, top_pol, top_imp]
    for stim in stims:
        stim_id = int(stim[0])
        style = stim[1].split(' ')[0]
        if style == 'Negative':
            idx = 0
        elif style == 'Positive':
            idx = 1
        elif style == 'Polite':
            idx = 2
        elif style == 'Impolite':
            idx = 3
        text = stim[3]
        cleaned_text = [clean_word(w) for w in stim[3].split(' ')]
        for word in cleaned_text:
            if word not in WORD_COUNTS:
                WORD_COUNTS[word] = 0
            WORD_COUNTS[word] += 1

        hb = tolist(stim[6])
        cptm = tolist(stim[-6])
        incon_ppl = tolist(stim[-1])[4:]
        ppl = tolist(stim[-2])[4:]
        label = stim[16]
        conf = stim[-3]
        # measures = [tolist(item) for item in lines[1:]] + [cptm, ppl, hb]  # remaining lines
        for i, measure in enumerate(metrics):
            hb = tolist(stim[6])
            cptm = tolist(stim[-6])
            incon_ppl = tolist(stim[-1])[4:]
            ppl = tolist(stim[-2])[4:]
            if measure in eye_measure_names:
                data, con, incon = get_statistic(df, stim_id, text, eye_measures[measure])
            else:
                if measure == 'hb':
                    data = hb
                if measure == 'cptm':
                    data = cptm
                if measure == 'ppl':
                    data = ppl
            if len(data) != len(cleaned_text):
                continue
            idx_to_move = find_move_indices(cleaned_text)
            move_indices([data], idx_to_move)
            for val, word in zip(data, cleaned_text):
                if word not in top_lists[idx][i]:
                    top_lists[idx][i][word] = []
                top_lists[idx][i][word].append(abs(val))

    for i, topn in enumerate(top_lists):
        top_lists[i] = clean_topn(topn)

    to_pop = []
    for w in WORD_COUNTS:
        if WORD_COUNTS[w] < 2:
            to_pop.append(w)
    for w in to_pop:
        WORD_COUNTS.pop(w)
    print('number of words is', len(WORD_COUNTS))
    print()

    n_vals = [25, 50, 75, 100, 125, 150, 200]
    correctness = []
    for n in n_vals:
        total = 0
        correct = [0 for _ in metrics]
        for stim in stims:
            total += 1
            style = stim[1].split(' ')[0]
            if style == 'Negative':
                top_n = top_lists[0]
                others = [top_lists[1], top_lists[2], top_lists[3]]
            elif style == 'Positive':
                top_n = top_lists[1]
                others = [top_lists[0], top_lists[2], top_lists[3]]
            elif style == 'Polite':
                top_n = top_lists[2]
                others = [top_lists[0], top_lists[1], top_lists[3]]
            elif style == 'Impolite':
                top_n == top_lists[3]
                others = [top_lists[0], top_lists[2], top_lists[1]]
            text = [clean_word(w) for w in stim[3].split(' ')]
            for i, measure in enumerate(metrics):
                for word in text:
                    top_words = top_n[i][:n]
                    if word in [w for w, _ in top_words]:
                        correct[i] += 1
                        break
                for word in text:
                    found = False
                    if found:
                        break
                    for other in others:
                        top_words = other[i][:n]
                        if word in [w for w, _ in top_words]:
                            correct[i] -= 1
                            found = True
                            break
        correctness.append(correct)
        print(correct)
        print(n)
        print(eye_measure_names)
        print(total)

for top_n, l in zip(top_lists, ['pos', 'neg', 'pol', 'imp']):
    with open(l + '.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(metrics)
        lines = [[] for _ in metrics]
        for i, measure in enumerate(metrics):
            for w, _ in top_n[i]:
                lines[i].append(w)
        for i in range(len(lines[0])):
            line = []
            for meas in lines:
                if i < len(meas):
                    line.append(meas[i])
                else:
                    line.append(' ')
            writer.writerow(line)


x = n_vals  # x axis for plot is n values
y = [[] for _ in metrics]
for correct in correctness:
    print(correct)
    for i, measure in enumerate(metrics):
        y[i].append(correct[i])
for i, measure in enumerate(metrics):
    plt.plot(x, y[i], label=measure)

plt.legend()
plt.show()
