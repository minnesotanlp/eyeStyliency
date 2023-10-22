import pandas as pd
import csv
from utils.utils import *


EYELINK_OUTPUT = 'data/IA_data.txt'
PROCESSED_OUTPUT_DESTINATION = 'data/IA_data.txt'  # filename of output for eyelink data with HAL_FREQ, LENGTH, WORD
# attributes added

# load data containing word frequencies in the stimuli's vocabulary
df = pd.read_csv('word_freq_data.csv')
df['Word'] = df['Word'].str.lower()

ppl_df = pd.read_csv('ppl.csv', sep=',', index_col=False)
eye_df = pd.read_csv(EYELINK_OUTPUT, sep='\t', index_col=False)


def get_ppl(stim_id, word):
    stim_id = int(stim_id)
    word = clean_word(word)
    entry = ppl_df.loc[ppl_df['stim_id'] == stim_id]
    if len(entry) == 0:
        return 'NA'
    ppls = entry['gpt2 ppl'].item().split(' ')
    text = entry['Text'].item()
    words = [clean_word(w) for w in text.split(' ')]
    if word == 'testuser':
        word = '<person>'
    elif word == '11:00_-':
        word = '-'
    elif word == 'mom' or word == 'plus' and stim_id == 10:
        word = 'mom\xa0plus'
    idx = words.index(word)
    return ppls[idx]


def has_prev_info(stim_id, ia_id, participant):
    ia_id = int(ia_id)
    stim_id = int(stim_id)
    info = eye_df.loc[(eye_df['stim_id'] == stim_id) & (eye_df['IA_ID'] == ia_id - 1) &
                      (eye_df['RECORDING_SESSION_LABEL'] == participant)]
    if len(info) == 0:
        return False
    has_prev = 0 if len(info['IA_FIRST_FIX_PROGRESSIVE']) == 0 else info['IA_FIRST_FIX_PROGRESSIVE'].tolist()[0]
    result = has_prev == "1"
    return result

prev_inc = False
prev_trial = 0
with open(EYELINK_OUTPUT) as f:
    lines = f.readlines()
    with open(PROCESSED_OUTPUT_DESTINATION, 'w') as out:
        # rewrite header line
        header = [s.strip() for s in lines[0].split('\t')]
        writer = csv.writer(out, delimiter='\t')
        print(header)
        congruent_idx = header.index('congruent')
        word_idx = header.index('IA_LABEL')
        stim_idx = header.index('stim_id')
        ia_idx = header.index('IA_ID')
        participant_idx = header.index('\ufeffRECORDING_SESSION_LABEL')
        writer.writerow(header + ['word', 'HAL_FREQ', 'LENGTH', 'ppl', 'has_prev'])
        # out.write(lines[0][:-1] + '\tword\tHAL_FREQ\tLENGTH\tppl\thas_prev\n')
        lines = lines[1:]

        extra = []
        for line in lines:
            items = [item.strip() for item in line.split('\t')]
            word = items[word_idx].lower().strip(' ,"\'.`\n?!@#¬†;():*')
            trial_index = int(items[1])
            if trial_index != prev_trial:
                skip_trial = prev_inc  # if prev trial was incongruent, discount this trial
            prev_inc = items[congruent_idx] != 'True'
            prev_trial = trial_index
            entry = df.loc[df['Word'] == word]
            if len(entry) == 0:  # word not found, use NA for freq
                freq = '.'
                length = str(len(word))
            else:
                length = entry['Length'].item()
                freq = entry['HAL LOG FREG'].item()
            if line[-1] == '\n':
                line = line[:-1]
            if skip_trial:
                continue

            ppl_val = get_ppl(items[stim_idx], word)
            has_prev = has_prev_info(items[stim_idx], items[ia_idx], items[participant_idx])

            line = items + [word, freq, length, ppl_val, has_prev]
            writer.writerow(line)

