from timeit import default_timer as timer
from datetime import timedelta
import pandas as pd
import numpy as np
from pathlib import Path
from datasets import ClassLabel, Features, Dataset, Value, Sequence
from transformers import (AutoModelForTokenClassification, TrainingArguments, Trainer)
import torch
device = torch.device("cuda")

from datasets import load_metric

# import custom utilities (path: tools/xlmroberta/utils.py)
from utils import tokenizer, tokenize_and_align_labels, data_collator

# Set Pathing

corpus = 'germeval2014'

p_corpus = Path.cwd() / 'corpora' / corpus
results_path = Path.cwd() / 'results' / f'roberta_{corpus}.csv'
model_dir = Path.cwd() / 'tools' / 'xlmroberta' / 'tmp' / corpus

# Load and prepare data

df_train = pd.read_feather(p_corpus / 'NER-de-train.feather')
df_test = pd.read_feather(p_corpus / 'NER-de-dev.feather')
df_validation = pd.read_feather(p_corpus / 'NER-de-test.feather')

labels_dict = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}

labels = ClassLabel(num_classes=len(labels_dict.keys()), names=list(labels_dict.keys()))
features = Features({'text': Sequence(Value(dtype="string")), 'labels': Sequence(labels)})

df_train['CoNLL_IOB2'] = df_train['CoNLL_IOB2'].replace(labels_dict)

df_train = df_train.groupby(['corpus', 'sentence_id'])[['token', 'CoNLL_IOB2']].agg(list)
df_train = df_train.rename(columns={'token': 'text', 'CoNLL_IOB2': 'labels'})

dataset_train = Dataset.from_pandas(df_train, features=features)

df_test['CoNLL_IOB2'] = df_test['CoNLL_IOB2'].replace(labels_dict)

df_test = df_test.groupby(['language', 'sentence_id'])[['token', 'CoNLL_IOB2']].agg(list)
df_test = df_test.rename(columns={'token': 'text', 'CoNLL_IOB2': 'labels'})

dataset_test = Dataset.from_pandas(df_test, features=features)

df_validation['CoNLL_IOB2'] = df_validation['CoNLL_IOB2'].replace(labels_dict)

df_validation = df_validation.groupby(['language', 'sentence_id'])[['token', 'CoNLL_IOB2']].agg(list)
df_validation = df_validation.rename(columns={'token': 'text', 'CoNLL_IOB2': 'labels'})

dataset_validation = Dataset.from_pandas(df_validation, features=features)

# Tokenize / Convert to XLM-Roberta tokens

label_list = dataset_train.features[f"labels"].feature.names

tokenized_train = dataset_train.map(tokenize_and_align_labels, batched=True)
tokenized_test = dataset_test.map(tokenize_and_align_labels, batched=True)
tokenized_validation = dataset_validation.map(tokenize_and_align_labels, batched=True)

# Set-up training parameters

learning_rate = 2e-05
batch_size = 8
epochs = 4

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

roberta = AutoModelForTokenClassification.from_pretrained("xlm-roberta-base", num_labels=len(label_list))
roberta.to(device)

training_args = TrainingArguments(
    output_dir=model_dir,
    evaluation_strategy="epoch",
    learning_rate=learning_rate,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    num_train_epochs=epochs,
    weight_decay=0.01,
)


trainer = Trainer(
    model=roberta,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# Train

start_training = timer()

trainer.train()

end_training = timer()
training_time = timedelta(seconds=end_training-start_training)

# Evaluate

trainer.eval_dataset = tokenized_validation
start_validation = timer()
results = trainer.evaluate()
end_validation = timer()
validation_time = timedelta(seconds=end_validation-start_validation)

# Save results

r = [{'task': key, **val} for key, val in results['eval_raw_results'].items() if type(val) == dict]

overall = {k.replace('overall_', ''): v for k, v in results['eval_raw_results'].items() if type(v) != dict}
overall['task'] = 'overall'

r.append(overall)

r = pd.DataFrame(r)

r['language'] = 'de'
r['training_duration'] = training_time.total_seconds()
r['validation_duration'] = validation_time.total_seconds()

r.to_csv(results_path, index=False)

import shutil

shutil.rmtree(model_dir)