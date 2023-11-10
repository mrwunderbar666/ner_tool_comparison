# Script to fine-tune XLM-RoBERTa with on a single corpus
# You can specify th size of the training data to be used.
# Either specify a percentage with `-t 0.2` =20% of training data
# Or an absolute number with `-t 1000` = use 1000 training samples  

# saves the trained model to tools/xlmroberta/models/model_id

# Uses the service https://wandb.ai for logging results
# If you wish to disable the service uncomment lines that menion "wandb"

# Takes following input arguments
#  -l LEARNING_RATE (float)         default 2e-5
#  -e EPOCHS (int)                  default 3
#  -b BATCH_SIZE (int)              default 8
#  -t TRAINING_SIZE (float)         Amount of training data to use.
#                                   If smaller than 1.0 uses a percentage of the training data.
#                                   If larger than 1.0 uses the absolute number.
#  -c CORPUS                        Name of the corpus
#  --languages                      Languages to use. E.g. --languages en de
#                                   Default is to use all languages

# Set-up training parameters
import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-l', '--learning_rate', type=float, default=2e-05,
                     help='learning rate', dest='learning_rate')
parser.add_argument('-e', '--epochs', type=int, default=3,
                     help='epochs', dest='epochs')
parser.add_argument('-b', '--batch_size', type=int, default=8,
                     help='batch size', dest='batch_size')
parser.add_argument('-t', '--training_size', type=float, default=1.0,
                     help='Percentage of training data to use', dest='training_size')
parser.add_argument('-c', '--corpus', type=str, default='conll',
                     help='Name of the corpus', dest='corpus')
parser.add_argument('--languages', nargs='+', required=False,
                    help="Languages to use. E.g. --languages en de", dest='languages')

args = parser.parse_args()


import wandb
wandb.init()
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
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Set Pathing
sys.path.insert(0, str(Path.cwd()))
# import custom utilities (path: tools/xlmroberta/utils.py)
from tools.xlmroberta.utils import (tokenizer, generate_tokenize_function, data_collator, 
                                    labels_dict, conll_labels, conll_features, compute_metrics)
from utils.registry import load_registry



learning_rate = args.learning_rate
epochs = args.epochs
batch_size = args.batch_size
corpus = args.corpus

if args.training_size > 1.0:
    training_size = int(args.training_size)
else:
    training_size = args.training_size

    print('Training Size:', training_size)
print('Corpus:', corpus)
if args.languages:
    print('Selected languages:', ", ".join(args.languages))
    model_dir = Path.cwd() / 'tools' / 'xlmroberta' / f'models_{corpus}_{"_".join(args.languages)}'
else:
    print('Using all languages')
    model_dir = Path.cwd() / 'tools' / 'xlmroberta' / f'models_{corpus}'

wandb.log({'training_size': training_size})


model_id = wandb.run.name

if not model_dir.exists():
    model_dir.mkdir(parents=True)

registry = load_registry()
if args.languages:
    df_corpora = registry.loc[(registry.language.isin(args.languages)) & (registry.corpus == corpus)]
else:
    df_corpora = registry.loc[registry.corpus == corpus]

datasets = {'train': [], 'test': [], 'validation': []}

tokenize = generate_tokenize_function("xlm-roberta-base", labels_dict)

# Load and prepare data
print('Preparing Data...')
for _, row in df_corpora.iterrows():
    if row['split'] not in datasets.keys():
        continue
    print('Preparing dataset:', row['path'])
    df = pd.read_feather(row['path'])
    df = df.loc[~df.token.isna(), ]
    df['CoNLL_IOB2'] = df['CoNLL_IOB2'].replace(labels_dict)
    df = df.groupby(['language', 'sentence_id'])[['token', 'CoNLL_IOB2']].agg(list)
    if row['split'] == 'train':
        if isinstance(training_size, int):
            df = df.sample(training_size)
        else:
            df = df.sample(frac=training_size)
    df = df.rename(columns={'token': 'text', 'CoNLL_IOB2': 'labels'})
    df = df.reset_index(drop=True)
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

tokenized_train = training_data.map(tokenize, batched=True)
tokenized_test = test_data.map(tokenize, batched=True)
tokenized_validation = validation_data.map(tokenize, batched=True)

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
    report_to="wandb",
    torch_compile=True
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
                'languages': args.languages, 
                'training_sentences': training_data.num_rows,
                'training_duration': training_time.total_seconds(),
                'validation_duration': validation_time.total_seconds(),
                'results': results
                }

print(model_details)
with open(model_path / 'model_infos.json', 'w') as f:
    json.dump(model_details, f)
