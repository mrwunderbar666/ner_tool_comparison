import os 
os.environ["WANDB_DISABLED"] = "true"
import sys
from timeit import default_timer as timer
from datetime import timedelta
import itertools
import pandas as pd
import numpy as np
from pathlib import Path
from datasets import ClassLabel, Features, Dataset, Value, Sequence
from transformers import (AutoTokenizer, DataCollatorForTokenClassification,
                          AutoModelForTokenClassification, TrainingArguments, Trainer)
import torch
device = torch.device("cuda")

from datasets import load_metric, concatenate_datasets

# Set Pathing
sys.path.insert(0, str(Path.cwd()))
# import custom utilities (path: tools/xlmroberta/utils.py)
from tools.xlmroberta.utils import (tokenizer, tokenize_and_align_labels, data_collator, 
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

learning_rate = 2e-05
batch_size = 8
epochs = 4

language_combinations = [
    ['en'], ['de'], ['es'], ['nl'], ['fr'], ['zh'], ['ar'], ['cs'], # every language individually
    ['en', 'de', 'es', 'nl', 'fr', 'zh', 'ar', 'cs'], # all languages together
    ['en', 'de', 'es', 'nl', 'fr', 'cs'] # only using latin scripts
]

model_dir = Path.cwd() / 'tools' / 'xlmroberta' / 'models'

if not model_dir.exists():
    model_dir.mkdir()

registry = load_registry()

model_id = 0
model_infos = []

for languages in language_combinations:

    print(f'Training languages: {" ".join(languages)}')

    corpora = registry.loc[registry.language.isin(languages)]

    unique_corpora = list(corpora.corpus.unique())

    corpora_combinations = []

    for i in range(1, len(unique_corpora) + 1):
        corpora_combinations += list(itertools.combinations(unique_corpora, i))

    print('total combinations of corpora:', len(corpora_combinations))

    datasets = {'train': [], 'test': []}

    for comb in corpora_combinations:

        model_id += 1

        print(f'Training combination: {" ".join(comb)}')

        corpora = registry.loc[(registry.language.isin(languages)) & (registry.corpus.isin(comb)) & (registry.split.isin(['train', 'test']))]

        # Load and prepare data
        for _, row in corpora.iterrows():
            df = pd.read_feather(row['path'])
            df = df.loc[~df.token.isna(), ]
            df['CoNLL_IOB2'] = df['CoNLL_IOB2'].replace(labels_dict)
            df = df.groupby(['language', 'sentence_id'])[['token', 'CoNLL_IOB2']].agg(list)
            df = df.rename(columns={'token': 'text', 'CoNLL_IOB2': 'labels'})
            datasets[row['split']].append(Dataset.from_pandas(df, features=conll_features))

        training_data = concatenate_datasets(datasets['train'])
        test_data = concatenate_datasets(datasets['test'])
        
        # dry run
        # training_data = training_data.shuffle()
        # training_data = training_data.select(list(range(1000)))
        # test_data = test_data.shuffle()
        # test_data = test_data.select(list(range(100)))

        # Tokenize / Convert to XLM-Roberta tokens

        label_list = training_data.features[f"labels"].feature.names

        tokenized_train = training_data.map(tokenize_and_align_labels, batched=True)
        tokenized_test = test_data.map(tokenize_and_align_labels, batched=True)

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

        trainer.save_model(model_path)

        model_details = {'model_id': model_id, 
                        'model_path': model_path, 
                        'batch_size': batch_size,
                        'epochs': epochs,
                        'learning_rate': learning_rate,
                        'languages': languages, 
                        'corpora': comb, 
                        'training_duration': training_time.total_seconds(),
                        }

        print(model_details)
        model_infos.append(model_details)


model_summaries = pd.DataFrame(model_infos)
model_summaries.to_csv(model_path / 'model_summaries.csv')