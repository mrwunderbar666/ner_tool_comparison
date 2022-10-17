from timeit import default_timer as timer
from datetime import timedelta
import pandas as pd
import numpy as np
from pathlib import Path
from datasets import ClassLabel, Features, Dataset, Value, Sequence
from transformers import (AutoTokenizer, DataCollatorForTokenClassification,
                          AutoModelForTokenClassification, TrainingArguments, Trainer)
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from datasets import load_metric

# import custom utilities (path: tools/xlmroberta/utils.py)
from utils import tokenizer, tokenize_and_align_labels, data_collator

# Set Pathing

corpus = 'conll'

p_corpus = Path.cwd() / 'corpora' / corpus
results_path = Path.cwd() / 'results' / f'roberta_{corpus}.csv'
model_dir = Path.cwd() / 'tools' / 'xlmroberta' / 'tmp' / corpus

# Load and prepare data

df_train_esp = pd.read_feather(p_corpus / 'esp.train.feather')
df_train_ned = pd.read_feather(p_corpus / 'ned.train.feather')
df_train_ned.sentence_id = df_train_ned.doc + '_' + df_train_ned.sentence_id.astype(str)
df_train_en = pd.read_feather(p_corpus / 'conll2003_en_train_iob.feather')

df_train = pd.concat([df_train_esp, df_train_en, df_train_ned])
df_train = df_train.loc[~df_train.token.isna(), ]

df_test_esp = pd.read_feather(p_corpus / 'esp.testa.feather')
df_test_ned = pd.read_feather(p_corpus / 'ned.testa.feather')
df_test_ned.sentence_id = df_test_ned.doc + '_' + df_test_ned.sentence_id.astype(str)

df_test_en = pd.read_feather(p_corpus / 'conll2003_en_test_iob.feather')

df_test = pd.concat([df_test_esp, df_test_ned, df_test_en])
df_test = df_test.loc[~df_test.token.isna(), ]


df_validation_esp = pd.read_feather(p_corpus / 'esp.testb.feather')
df_validation_ned = pd.read_feather(p_corpus / 'ned.testb.feather')
df_validation_ned.sentence_id = df_validation_ned.doc + '_' + df_validation_ned.sentence_id.astype(str)

df_validation_en = pd.read_feather(p_corpus / 'conll2003_en_validation_iob.feather')

df_validation = pd.concat([df_validation_esp, df_validation_ned, df_validation_en])
df_validation = df_validation.loc[~df_validation.token.isna(), ]


labels_dict = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}

labels = ClassLabel(num_classes=len(labels_dict.keys()), names=list(labels_dict.keys()))
features = Features({'text': Sequence(Value(dtype="string")), 'labels': Sequence(labels)})

df_train['IOB2'] = df_train['IOB2'].replace(labels_dict)

df_train = df_train.groupby(['language', 'sentence_id'])[['token', 'IOB2']].agg(list)
df_train = df_train.rename(columns={'token': 'text', 'IOB2': 'labels'})

dataset_train = Dataset.from_pandas(df_train, features=features)

df_test['IOB2'] = df_test['IOB2'].replace(labels_dict)

df_test = df_test.groupby(['language', 'sentence_id'])[['token', 'IOB2']].agg(list)
df_test = df_test.rename(columns={'token': 'text', 'IOB2': 'labels'})

dataset_test = Dataset.from_pandas(df_test, features=features)

df_validation['IOB2'] = df_validation['IOB2'].replace(labels_dict)


validation_sets = {}

for language in df_validation.language.unique():
    tmp_df = df_validation.loc[df_validation.language == language, ]
    tmp_df = tmp_df.groupby(['corpus', 'sentence_id'])[['token', 'IOB2']].agg(list)
    tmp_df = tmp_df.rename(columns={'token': 'text', 'IOB2': 'labels'})
    validation_sets[language] = Dataset.from_pandas(tmp_df, features=features)

# Tokenize / Convert to XLM-Roberta tokens

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["text"], truncation=True, is_split_into_words=True)
    labels = []
    for i, label in enumerate(examples[f"labels"]):
        # Map tokens to their respective word.
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:  # Set the special tokens to -100.
            if word_idx is None:
                label_ids.append(-100)
            # Only label the first token of a given word.
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs

label_list = dataset_train.features[f"labels"].feature.names


tokenized_train = dataset_train.map(tokenize_and_align_labels, batched=True)
tokenized_test = dataset_test.map(tokenize_and_align_labels, batched=True)
tokenized_validation_sets = {lang: ds.map(tokenize_and_align_labels, batched=True) for lang, ds in validation_sets.items()}

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

evaluations = []

for language, validation_set in tokenized_validation_sets.items():

    trainer.eval_dataset = validation_set
    
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
    r['language'] = language
    r['training_duration'] = training_time.total_seconds()
    r['validation_duration'] = validation_time.total_seconds()
    evaluations.append(r)

results_df = pd.concat(evaluations)

results_df.to_csv(results_path, index=False)


import shutil

shutil.rmtree(model_dir)