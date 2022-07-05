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
from datasets import Dataset
from transformers import (AutoTokenizer,
                          AutoModelForTokenClassification, TrainingArguments, Trainer)
import torch
device = torch.device("cuda")

from datasets import load_metric, concatenate_datasets

# Set Pathing
sys.path.insert(0, str(Path.cwd()))
# import custom utilities (path: tools/xlmroberta/utils.py)
from tools.xlmroberta.utils import (get_combination, tokenizer, tokenize_and_align_labels, data_collator, 
                                    labels_dict, conll_labels, conll_features)
from utils.registry import load_registry

# set-up tokenizer
print('Loading tokenizer...')
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

print('Loading metric...')

metric = load_metric("seqeval")

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
        'raw_results': results
    }


# Set-up training parameters

parser = argparse.ArgumentParser()

parser.add_argument('-l', '--learning_rate', type=float, default=2e-05,
                     help='learning rate', dest='learning_rate')
parser.add_argument('-e', '--epochs', type=int, default=3,
                     help='epochs', dest='epochs')
parser.add_argument('-b', '--batch_size', type=int, default=8,
                     help='batch size', dest='batch_size')
parser.add_argument('-m', '--model_combination', type=int, default=0,
                     help='which language combindation to use', dest='model_combination')

args = parser.parse_args()


learning_rate = args.learning_rate
epochs = args.epochs
batch_size = args.batch_size
model_id = args.model_combination
languages, corpora = get_combination(model_id)

print('Model ID:', model_id)
print('Combination:', languages, corpora)

wandb.log({'model_id': model_id, 'languages': languages, 'corpora': corpora})

model_dir = Path.cwd() / 'tools' / 'xlmroberta' / 'models'

if not model_dir.exists():
    model_dir.mkdir()

registry = load_registry()
registry = registry.loc[~(registry.corpus == 'wikiann')]
df_corpora = registry.loc[(registry.language.isin(languages)) & (registry.corpus.isin(corpora))]

datasets = {'train': [], 'test': [], 'validation': []}

# Load and prepare data
for _, row in df_corpora.iterrows():
    if row['split'] not in datasets.keys():
        continue
    df = pd.read_feather(row['path'])
    df = df.loc[~df.token.isna(), ]
    df['CoNLL_IOB2'] = df['CoNLL_IOB2'].replace(labels_dict)
    df = df.groupby(['language', 'sentence_id'])[['token', 'CoNLL_IOB2']].agg(list)
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
                'corpora': corpora, 
                'training_duration': training_time.total_seconds(),
                'validation_duration': validation_time.total_seconds(),
                'results': results
                }

print(model_details)
with open(model_path / 'model_infos.json', 'w') as f:
    json.dump(model_details, f)
