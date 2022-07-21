import wandb
wandb.init()
import argparse
import sys
import json
from timeit import default_timer as timer
from datetime import timedelta
import pandas as pd
import numpy as np
from pathlib import Path
from datasets import Dataset, concatenate_datasets
from transformers import (AutoModelForTokenClassification, TrainingArguments, Trainer)
import torch
device = torch.device("cuda")


# Set Pathing
sys.path.insert(0, str(Path.cwd()))
# import custom utilities (path: tools/xlmroberta/utils.py)
from tools.xlmroberta.utils import (get_combination, tokenizer, tokenize_and_align_labels, data_collator, 
                                    labels_dict, conll_labels, conll_features, compute_metrics)
from utils.registry import load_registry


# Set-up training parameters

parser = argparse.ArgumentParser()

parser.add_argument('-l', '--learning_rate', type=float, default=2e-05,
                     help='learning rate', dest='learning_rate')
parser.add_argument('-e', '--epochs', type=int, default=3,
                     help='epochs', dest='epochs')
parser.add_argument('-b', '--batch_size', type=int, default=8,
                     help='batch size', dest='batch_size')
parser.add_argument('-t', '--training_size', type=int, default=1000,
                     help='Samples of training data to use per corpus', dest='training_size')

args = parser.parse_args()


learning_rate = args.learning_rate
epochs = args.epochs
batch_size = args.batch_size
training_size = args.training_size
languages = ['en', 'de', 'es', 'nl', 'fr', 'cs']

print('Training Size:', training_size)

wandb.log({'training_size': training_size})

model_dir = Path.cwd() / 'tools' / 'xlmroberta' / 'models_varying_b'

model_id = wandb.run.name

if not model_dir.exists():
    model_dir.mkdir()

registry = load_registry()
df_corpora = registry.loc[(registry.language.isin(languages)) & (registry.corpus != 'wikiann') & (registry.corpus != 'europeana') & (registry.corpus != 'emerging')]

datasets = {'train': [], 'test': [], 'validation': []}

# Load and prepare data
print('Preparing Data...')
for _, row in df_corpora.iterrows():
    if row['split'] not in datasets.keys():
        continue
    if row['split'] != 'train' and row['corpus'] != 'conll':
        continue
    print('Preparing dataset:', row['path'])
    df = pd.read_feather(row['path'])
    df = df.loc[~df.token.isna(), ]
    df['CoNLL_IOB2'] = df['CoNLL_IOB2'].replace(labels_dict)
    df = df.groupby(['language', 'sentence_id'])[['token', 'CoNLL_IOB2']].agg(list)
    if row['split'] == 'train':
        df = df.sample(n=training_size)
    df = df.rename(columns={'token': 'text', 'CoNLL_IOB2': 'labels'})
    datasets[row['split']].append(Dataset.from_pandas(df, features=conll_features))

training_data = concatenate_datasets(datasets['train'])
test_data = concatenate_datasets(datasets['test'])
validation_data = concatenate_datasets(datasets['validation'])

# dry run
# training_data = training_data.shuffle()
# training_data = training_data.select(list(range(1000)))
# test_data = test_data.shuffle()
# test_data = test_data.select(list(range(100)))

# Tokenize / Convert to XLM-Roberta tokens

label_list = training_data.features[f"labels"].feature.names

tokenized_train = training_data.map(tokenize_and_align_labels, batched=True)
tokenized_test = test_data.map(tokenize_and_align_labels, batched=True)
tokenized_validation = validation_data.map(tokenize_and_align_labels, batched=True)

roberta = AutoModelForTokenClassification.from_pretrained("xlm-roberta-base", num_labels=len(label_list))
roberta.to(device)

model_path = model_dir / str(model_id)
if not model_path.exists():
    model_path.mkdir()

training_args = TrainingArguments(
    output_dir=model_path,
    evaluation_strategy="epoch",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    weight_decay=0.01,
    save_steps=2000,
    save_total_limit=1, # only save 1 checkpoint,
    report_to="wandb"
)

trainer = Trainer(
    model=roberta,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train

start_training = timer()
trainer.train()
end_training = timer()
training_time = timedelta(seconds=end_training-start_training)

trainer.eval_dataset = tokenized_validation
start_validation = timer()
results = trainer.evaluate()
end_validation = timer()
validation_time = timedelta(seconds=end_validation-start_validation)

trainer.save_model(model_path)

model_details = {'model_id': model_id, 
                'model_path': str(model_path), 
                'batch_size': batch_size,
                'epochs': epochs,
                'learning_rate': learning_rate,
                'languages': languages, 
                'training_sentences': training_data.num_rows,
                'training_duration': training_time.total_seconds(),
                'validation_duration': validation_time.total_seconds(),
                'results': results
                }

print(model_details)
with open(model_path / 'model_infos.json', 'w') as f:
    json.dump(model_details, f)
