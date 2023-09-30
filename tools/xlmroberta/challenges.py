import sys
from pathlib import Path

import pandas as pd
import numpy as np

import torch
from transformers import AutoModelForTokenClassification
from tqdm import tqdm

# Set Pathing
sys.path.insert(0, str(Path.cwd()))
# import custom utilities (path: tools/xlmroberta/utils.py)
from tools.xlmroberta.utils import (tokenizer, labels_dict, get_model_id_with_full_trainingdata)
from utils.challenges import load_challenges

# flip labels_dict
labels_dict = {v: k for k, v in labels_dict.items()}


challenges = load_challenges()

challenges['tool'] = 'xlmroberta'
challenges['tokens'] = ''
challenges['iob'] = ''

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_id = get_model_id_with_full_trainingdata() # model trained with all corpora / languages

p = Path.cwd()
model = p / 'tools' / 'xlmroberta' / 'models' / str(model_id)
results_path = Path.cwd() / 'results' / 'xlmroberta_challenges.json'

roberta = AutoModelForTokenClassification.from_pretrained(model)
roberta.to(device)

print('Running challenges...')

with tqdm(total=len(challenges), unit="sentence") as pbar:
    for index, row in challenges.iterrows():

        inputs = tokenizer(row['text'], padding=True, return_tensors='pt')
        inputs = inputs.to(device)
        outputs = roberta(**inputs)
        predictions = outputs.logits.cpu()
        predictions = predictions.detach().numpy()
        predictions = np.argmax(predictions, axis=2)

        challenges.at[index, 'tokens'] = inputs.tokens()
        challenges.at[index, 'iob'] = [labels_dict[p] for p in predictions[0]]
        pbar.update(1)

challenges.to_json(results_path, orient="records")

print('Done!')