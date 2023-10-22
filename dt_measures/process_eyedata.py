import collections
from nltk.corpus import stopwords

EYEDATA_FILE = 'appended_816.txt'

# coef from LMM (from R)
coefs = {"10e": [200.0972, 241.7376, 239.194],
         "11a": [279.9549, 277.9744, 285.443],
         "12c": [320.2948, 249.7709, 301.3392],
         "13c": [214.1819, 219.7105, 233.7532],
         "15d": [305.3047, 208.3561, 243.8766],
         "16a": [237.3665, 255.1506, 273.2892],
         "17b": [245.1925, 230.8326, 260.0104],
         "18c": [255.1647, 246.526, 253.9547],
         "19d": [183.9415, 248.4709, 255.8281],
         "5a": [205.802, 216.8527, 213.3133],
         "6b": [191.2242, 199.3068, 201.3174],
         "7c": [246.4934, 200.8828, 227.8702],
         "9d": [261.7995, 221.3832, 221.9724]}

# hal, len coef for DT, FFD, FRD
HL = [(-12.87296, 32.17595), (-3.107074, -0.9392005), (-5.024487, 6.504597)]

def calc_av(l):
    if len(l) == 0:
        return 0
    return round(sum(l) / len(l), 2)


def compare_ias(exp, control):
    if not exp or not control:
        return 0
    return round(calc_av(exp) - calc_av(control), 2)


# measures
measures = {'FFD': 1,
            'Dwell Time': 0,
            'FRD': 2}


def parse_item(item):
    subj = items[0]
    dwell = items[2]
    ffd = items[4]
    frd = items[5]
    iaid = int(items[10])
    ialabel = items[11]
    stim_id = int(items[20])
    halfreq = items[26]
    length = items[27]
    word = items[25]
    return subj, dwell, ffd, frd, iaid, ialabel, stim_id, halfreq, length, word

for measure, stat_idx in measures.items():
    # scores for DT
    stim_mapc = {}  # for congruent stimuli
    stim_mapi = {}  # for incongruent stimuli
    for i in range(0, 89):
        stim_mapi[i] = {}
        stim_mapc[i] = {}

    with open(EYEDATA_FILE) as f:
        lines = f.readlines()
        lines = lines[1:]  # ignore header

    # create stim maps
    leftover_stat = 0
    for line in lines:
        items = line.split('\t')
        if items[19] == "True":
            stim_map = stim_mapc
        else:
            stim_map = stim_mapi
        subj, dwell, ffd, frd, iaid, ialabel, stim_id, halfreq, length, word = parse_item(items)
        measures = [dwell, ffd, frd]
        if halfreq == '.':
            halfreq = 6

        # set up stim map
        if stim_id not in stim_map:
            stim_map[stim_id] = {}
        if iaid not in stim_map[stim_id]:
            stim_map[stim_id][iaid] = []
        stat = measures[stat_idx]
        if stat == '.':
            stim_map[stim_id][iaid].append(('.', ialabel))
            continue

        # calculate statistic
        stat = float(stat)
        expected = coefs[subj][stat_idx] + HL[stat_idx][0] * float(halfreq) + HL[stat_idx][1] * float(length)
        diff = stat - expected
        stim_map[stim_id][iaid].append((stat, ialabel))

    with open(measure + '_subresults.txt', 'w') as sub, \
            open(measure + '_conresults.txt', 'w') as con, \
            open(measure + '_incresults.txt', 'w') as inc:
        for stim, ias in stim_mapi.items():
            experimental = [0 for _ in ias]
            diffs = [0 for _ in ias]
            controls = [0 for _ in ias]
            words = []
            for scores in ias.values():
                words.append(scores[0][1])

            for i, (ia, scores) in enumerate(ias.items()):
                numerical = [n for n, _ in scores if n != '.']
                control = [n for n, _ in stim_mapc[stim][ia] if n != '.']
                diff = compare_ias(numerical, control)
                if words[i] in stopwords.words('english') and diff > 0:
                    j = i
                    left_idx, right_idx = None, None
                    while j >= 0:
                        j -= 1
                        if words[j] not in stopwords.words('english'):
                            left_idx = j
                            break
                    j = i
                    while j < len(words):
                        if words[j] not in stopwords.words('english'):
                            right_idx = j
                        j += 1

                    if left_idx and right_idx:
                        idx = left_idx if i - left_idx < i - right_idx else right_idx  # right wins in case of tie
                    else:
                        idx = left_idx if left_idx else right_idx

                else:
                    idx = i
                diffs[idx] += max(diff, 0)
                experimental[idx] += calc_av(numerical)
                controls[idx] += calc_av(control)

            for e, c, d in zip(experimental, controls, diffs):
                sub.write(str(d) + ' ')
                con.write(str(c) + ' ')
                inc.write(str(e) + ' ')
            sub.write('\n')
            con.write('\n')
            inc.write('\n')
