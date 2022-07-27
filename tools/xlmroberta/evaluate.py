import sys
import json
from pathlib import Path
from timeit import default_timer as timer
from datetime import timedelta

import pandas as pd

import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer
from datasets import Dataset

# Set Pathing
sys.path.insert(0, str(Path.cwd()))
# import custom utilities (path: tools/xlmroberta/utils.py)
from tools.xlmroberta.utils import (get_combination, tokenizer, tokenize_and_align_labels, data_collator, 
                                    labels_dict, conll_labels, conll_features, compute_metrics)
from utils.registry import load_registry


registry = load_registry()

languages = ['en', 'de', 'es', 'nl', 'fr', 'zh', 'ar', 'cs', 'it', 'hu']

df_corpora = registry.loc[(registry.split == 'validation') & (registry.language.isin(languages))]

validation_sets = {}

# Load and prepare data
for _, row in df_corpora.iterrows():
    df = pd.read_feather(row['path'])
    df = df.loc[~df.token.isna(), ]
    df['CoNLL_IOB2'] = df['CoNLL_IOB2'].replace(labels_dict)
    df = df.groupby(['language', 'sentence_id'])[['token', 'CoNLL_IOB2']].agg(list)
    df = df.rename(columns={'token': 'text', 'CoNLL_IOB2': 'labels'})
    ds = Dataset.from_pandas(df, features=conll_features)
    ds = ds.map(tokenize_and_align_labels, batched=True)
    validation_sets[row['path']] = ds

device = torch.device("cuda")

p = Path.cwd()
models = p.glob('tools/xlmroberta/models/*')

for m in models:

    model_infos = m / 'model_infos.json'
    if not model_infos.exists():
        print('MODEL INFO DOES NOT EXIST', model_infos)
        continue

    results_destination = m / 'eval_results.csv'
    if results_destination.exists():
        print('Model already evaluated', results_destination)
        continue

    with open(m / 'model_infos.json') as f:
        infos = json.load(f)

    print('Loading model', infos['model_id'])
    print(infos['model_path'])

    roberta = AutoModelForTokenClassification.from_pretrained(m)
    roberta.to(device)

    roberta.eval()

    trainer = Trainer(
        model=roberta,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    end_results = []

    for v_path, v in validation_sets.items():
        trainer.eval_dataset = v
        start_validation = timer()
        results = trainer.evaluate()
        end_validation = timer()
        validation_time = timedelta(seconds=end_validation-start_validation)
        r = [{'task': key, **val} for key, val in results['eval_raw_results'].items() if type(val) == dict]
        overall = {k.replace('overall_', ''): v for k, v in results['eval_raw_results'].items() if type(v) != dict}
        overall['task'] = 'overall'
        r.append(overall)
        r = pd.DataFrame(r)
        r['validation_corpus'] = v_path
        r['validation_duration'] = validation_time.total_seconds()
        r['model_id'] = infos['model_id']
        r['model_languages'] = ", ".join(infos['languages'])
        r['model_corpora'] = ", ".join(infos['corpora'])
        end_results.append(r)

    eval_results = pd.concat(end_results)
    eval_results.to_csv(m / 'eval_results.csv', index=False)

    del trainer
    del roberta
    torch.cuda.empty_cache()

    print(80 * '-', '\n')

print('Done!')