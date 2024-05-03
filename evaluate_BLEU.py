from nltk.translate.bleu_score import corpus_bleu
import torch

from model.T5 import T5Model_Pretrained

import json
import os
import sys

with open(f"./config/{sys.argv[1]}.json") as f:
    CONFIG = json.load(f)


model = T5Model_Pretrained("data/networkx/pretrained/fine_tuned_model.colab.1702232161.9665232.3.python")

dataset = []

jsons = [f"{CONFIG["DataStorePath"]}/{f}" for f in os.listdir(CONFIG["DataStorePath"])]
for js in jsons:
    if js.endswith(".json"):
        with open(js) as f:
            data = json.load(f)
            for pair in data:
                dataset.append(pair)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.model.to(device)

size = len(dataset)
print(size)
inputs = [data["code"] for data in dataset]
expected_output = [data["docstring"].split() for data in dataset]
calculated_output = [model.predict(s).split() for s in inputs]
calculated_output = []
for i,s in inputs:
    calc = model.predict(s, device).split()
    print(f"{i}/{size}:{calc[:5]}", end="\r")

bleu_score = corpus_bleu(expected_output, calculated_output)

print(bleu_score)