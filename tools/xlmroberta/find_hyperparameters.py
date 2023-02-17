# Script for finding optimal hyperparameters
# Uses the CoNLL++ dataset https://huggingface.co/datasets/conllpp

# Depends on service by wandb (https://wandb.ai) 
# for automatic sweep controls and result logging

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('-l', '--learning_rate', type=float, default=5e-5, metavar='N',
                     help='learning rate', dest='learning_rate')
parser.add_argument('-e', '--epochs', type=int, default=2, metavar='N',
                     help='epochs', dest='epochs')
parser.add_argument('-b', '--batch_size', type=int, default=8, metavar='N',
                     help='batch size', dest='batch_size')

args = parser.parse_args()

import wandb

import sys
from pathlib import Path

# Set Pathing
sys.path.insert(0, str(Path.cwd()))
# import custom utilities (path: tools/xlmroberta/utils.py)
from tools.xlmroberta.utils import (tokenizer, tokenize_and_align_labels, data_collator, 
                                   compute_metrics)

from datasets import load_dataset
from transformers import (AutoModelForTokenClassification, TrainingArguments, Trainer)
import torch

wandb.init()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


learning_rate = args.learning_rate
epochs = args.epochs
batch_size = args.batch_size

print(f'learning rate: {learning_rate}')
print(f'epochs: {epochs}')
print(f'batch_size: {batch_size}')


conll = load_dataset("conllpp")

label_list = conll["train"].features[f"ner_tags"].feature.names


tokenized_conll = conll.map(tokenize_and_align_labels, batched=True)

roberta = AutoModelForTokenClassification.from_pretrained("xlm-roberta-base", num_labels=len(label_list))
roberta.to(device)


training_args = TrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    save_steps=2000,
    save_total_limit=1, # only save 1 checkpoint,
    report_to="wandb",
)

trainer = Trainer(
    model=roberta,
    args=training_args,
    train_dataset=tokenized_conll["train"],
    eval_dataset=tokenized_conll["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

trainer.train()