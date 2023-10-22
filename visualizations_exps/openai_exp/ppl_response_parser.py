import json
import csv
import numpy as np
import pandas as pd

RESPONSE_FILE = 'gpt3_ppl_responses.csv'
df = pd.read_csv('ppl.csv', index_col=False)


def get_logprobs(response_dict):
    logprobs = response_dict['choices'][0]["logprobs"]
    return zip(logprobs['tokens'], logprobs['token_logprobs'])


def get_text(stim_id):
    return df.loc[df['id'] == stim_id]['Text'].item()


def token_to_word_probs(token_logprobs, text):
    words = text.split(' ')
    scores = [0 for _ in words]
    curr_word = 0
    for token, logprob in token_logprobs:
        token = token.strip(' ')
        if logprob:
            scores[curr_word] += np.exp(logprob)
        if words[curr_word].endswith(token):
            curr_word += 1
    return scores


def list_to_str(l):
    return ' '.join([str(i) for i in l])


with open(RESPONSE_FILE) as f:
    reader = csv.reader(f)
    lines = list(reader)
    header = lines[0]

    gpt3_ppl = []
    for line in lines[1:]:
        stim_id = line[0]
        response = line[1]
        response_dict = json.loads(response)
        text = get_text(int(stim_id))
        token_logprobs = get_logprobs(response_dict)
        word_logprobs = token_to_word_probs(token_logprobs, text)
        gpt3_ppl.append(word_logprobs)
    df.insert(len(df.columns), 'gpt3 ppl', [list_to_str(ppl) for ppl in gpt3_ppl])
    df.to_csv('ppl.csv')
