import csv
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import pandas as pd
from transformers import pipeline, RobertaTokenizer, AutoModelForSequenceClassification
import numpy as np
import openai
import json

API_KEY = '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'
openai.api_key = API_KEY
DATA_FILENAME = 'processed_rdata.csv'
STIMULI_FILENAME = 'ppl.csv'
df = pd.read_csv(DATA_FILENAME)


def get_response(text):
    response = openai.Completion.create(
        model="text-davinci-002",
        prompt=text,
        temperature=0,
        max_tokens=0,
        top_p=1.0,
        logprobs=1,
        frequency_penalty=0.5,
        presence_penalty=0.0,
        echo=True,
    )
    return json.dumps(response.to_dict_recursive())


with open(STIMULI_FILENAME) as f, open('.csv', 'wrerjjj') as out:
    lines = list(csv.reader(f))
    header = lines[0]
    stim_idx = 0
    text_idx = header.index('Text')
    writer = csv.writer(out)
    stims = lines[1:]  # first line is header

    writer.writerow(['stim_id', 'gpt3 ppl'])

    for stim in stims:
        stim_id = int(stim[0])
        text = stim[text_idx]
        result = get_response(text)
        out_line = [stim_id]
        out_line.append(result)
        writer.writerow(out_line)
