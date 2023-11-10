import sys
import json
from pathlib import Path
from timeit import default_timer as timer
from datetime import timedelta

import pandas as pd
from argparse import ArgumentParser

import torch
from transformers import AutoModelForTokenClassification, Trainer
from datasets import Dataset

# Set Pathing
sys.path.insert(0, str(Path.cwd()))
# import custom utilities (path: tools/xlmroberta/utils.py)
from tools.xlmroberta.utils import (get_combination, tokenizer, generate_tokenize_function, data_collator, 
                                    labels_dict, conll_labels, conll_features, compute_metrics, get_model_id_with_full_trainingdata)
from utils.registry import load_registry


argparser = ArgumentParser(prog='Run XLM-RoBERTa Evaluation')
argparser.add_argument('--debug', action='store_true', help='Debug flag (only test a random sample)')
args = argparser.parse_args()

tokenize = generate_tokenize_function("xlm-roberta-base", labels_dict)

registry = load_registry()

languages = ['en', 'de', 'es', 'nl', 'fr', 'zh', 'ar', 'cs', 'it', 'hu']

df_corpora = registry.loc[(registry.split == 'validation') & (registry.language.isin(languages))]

validation_sets = {}

# Load and prepare data
for _, row in df_corpora.iterrows():
    df = pd.read_feather(row['path'])
    df = df.loc[~df.token.isna(), :]
    if args.debug:
        import random
        sample_size = min(len(df.sentence_id.unique().tolist()), 100)
        sentence_ids = random.sample(df.sentence_id.unique().tolist(), sample_size)
        df = df.loc[df.sentence_id.isin(sentence_ids), :]

    df['CoNLL_IOB2'] = df['CoNLL_IOB2'].replace(labels_dict)
    df = df.groupby(['sentence_id'])[['token', 'CoNLL_IOB2']].agg(list)
    df = df.rename(columns={'token': 'text', 'CoNLL_IOB2': 'labels'})
    ds = Dataset.from_pandas(df, features=conll_features)
    ds = ds.map(tokenize, batched=True)
    validation_sets[row['path']] = {'dataset': ds, 
                                    'language': row['language'],
                                    'corpus': row['corpus'],
                                    'subset': row['subset'],
                                    'tokens': row['tokens'],
                                    'sentences': row['sentences']}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

p = Path.cwd()
model_id = get_model_id_with_full_trainingdata() # use the model with full training set

model_path = p / 'tools' / 'xlmroberta' / 'models' / str(model_id) 
model_infos = model_path / 'model_infos.json'
results_destination = p / 'results' / 'xlmroberta.csv'

if not model_infos.exists():
    print('MODEL INFO DOES NOT EXIST', model_infos)
    sys.exit(1)


with open(model_infos) as f:
    infos = json.load(f)

print('Loading model', infos['model_id'])
print(infos['model_path'])

roberta = AutoModelForTokenClassification.from_pretrained(model_path)
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
    trainer.eval_dataset = v['dataset']
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
    r['language'] = v['language']
    r['corpus'] = v['corpus']
    r['subset'] = v['subset']
    r['tokens'] = v['tokens']
    r['sentences'] = v['sentences']
    r['model_id'] = infos['model_id']
    r['model_languages'] = ", ".join(infos['languages'])
    r['model_corpora'] = ", ".join(infos['corpora'])
    end_results.append(r)

eval_results = pd.concat(end_results)
eval_results.to_csv(results_destination, index=False)

del trainer
del roberta
torch.cuda.empty_cache()

print(80 * '-', '\n')

print('Done!')