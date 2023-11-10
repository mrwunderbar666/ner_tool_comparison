import itertools
import typing
from pathlib import Path
from transformers import AutoTokenizer, DataCollatorForTokenClassification
from transformers.models.roberta.tokenization_roberta_fast import RobertaTokenizerFast

from datasets import ClassLabel, Features, Value, Sequence
from utils.registry import load_registry
import pandas as pd
import numpy as np
import sys

sys.path.insert(0, str(Path.cwd()))
from utils.metrics import compute_metrics as _compute_metrics

labels_dict = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}

conll_labels = ClassLabel(num_classes=len(labels_dict.keys()), names=list(labels_dict.keys()))
conll_features = Features({'text': Sequence(Value(dtype="string")), 'labels': Sequence(conll_labels)})
label_list = conll_features['labels'].feature.names

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

def generate_tokenize_function(model, labels_dict: dict, 
                               text_col="text", 
                               label_col="labels",
                               max_length=128) -> typing.Callable:
    """
        This function loads the tokenizer for the BERT model and 
        aligns the NE labels to match the BERT-subtokens
        It returns a tokenize function that can be applied to the dataset
    """
    tokenizer = get_tokenizer(model)
    tokenizer.add_prefix_space = isinstance(tokenizer, RobertaTokenizerFast)

    # 'B-PER' => 'I-PER'`
    b_to_i_label = {k: "" for k in labels_dict if k.startswith('B')}
    for k in labels_dict:
        if k.startswith('I-'):
            b_to_i_label['B-' + k.replace('I-', '')] = k

    b_to_i_label = {labels_dict[k]: labels_dict[v] for k,v in b_to_i_label.items()}
    b_to_i_label[0] = 0

    def _tokenize_func(examples, truncation=True, 
                       is_split_into_words=True,
                       padding="max_length"):
        tokenized_inputs = tokenizer(
            examples[text_col],
            padding=padding,
            truncation=truncation,
            is_split_into_words=is_split_into_words,
            max_length=max_length)
        labels = []
        for i, label in enumerate(examples[label_col]):
            # Map tokens to their respective word.
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label
                else:
                    label_ids.append(b_to_i_label[label[word_idx]])
            previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs[label_col] = labels
        return tokenized_inputs
    return _tokenize_func


def get_tokenizer(model: str):
    return AutoTokenizer.from_pretrained(model)

def generate_tokenize_function(model: str, 
                               labels_dict: dict, 
                               text_col="text", 
                               label_col="labels",
                               max_length=512):
    """
        This function loads the tokenizer for the BERT model and 
        aligns the NE labels to match the BERT-subtokens
        It returns a tokenize function that can be applied to the dataset
    """
    tokenizer = get_tokenizer(model)
    tokenizer.add_prefix_space = isinstance(tokenizer, RobertaTokenizerFast)

    # 'B-PER' => 'I-PER'`
    b_to_i_label = {k: "" for k in labels_dict if k.startswith('B')}
    for k in labels_dict:
        if k.startswith('I-'):
            b_to_i_label['B-' + k.replace('I-', '')] = k

    b_to_i_label = {labels_dict[k]: labels_dict[v] for k,v in b_to_i_label.items()}
    b_to_i_label[0] = 0

    def _tokenize_func(examples, truncation=True, 
                       is_split_into_words=True,
                       padding="max_length"):
        tokenized_inputs = tokenizer(
            examples[text_col],
            padding=padding,
            truncation=truncation,
            is_split_into_words=is_split_into_words,
            max_length=max_length)
        labels = []
        for i, label in enumerate(examples[label_col]):
            # Map tokens to their respective word.
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            for word_idx in word_ids:
                # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                # ignored in the loss function.
                if word_idx is None:
                    label_ids.append(-100)
                # We set the label for the first token of each word.
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                # For the other tokens in a word, we set the label to either the current label
                else:
                    label_ids.append(b_to_i_label[label[word_idx]])
            previous_word_idx = word_idx
            labels.append(label_ids)
        tokenized_inputs[label_col] = labels
        return tokenized_inputs
    return _tokenize_func


def generate_combinations() -> pd.DataFrame:

    language_combinations = [
        # every language individually
        ['en'], ['de'], ['es'], ['nl'], ['fr'], ['zh'], ['ar'], ['cs'], 
        ['pt'], ['it'], ['hu'], ['ca'],
        # all languages together
        ['en', 'de', 'es', 'nl', 'fr', 'zh', 'ar', 'cs', 'pt', 'it', 'hu', 'ca'], 
        # only using latin scripts
        ['en', 'de', 'es', 'nl', 'fr', 'cs', 'pt', 'hu', 'it', 'ca'] 
    ]

    registry = load_registry()
    # registry = registry.loc[~(registry.corpus == 'wikiann')]

    model_combinations = []
    for languages in language_combinations:
        corpora = registry.loc[registry.language.isin(languages)]
        unique_corpora = list(corpora.corpus.unique())
        for i in range(1, len(unique_corpora) + 1):
            combs = list(itertools.combinations(unique_corpora, i))
            for c in combs:
                if len(registry.loc[(registry.language.isin(languages)) & (registry.corpus.isin(c))]) > 0:
                    # l = ", ".join(languages)
                    c = list(c)
                    c.sort()
                    model_combinations.append({'languages': languages, 'corpora': c})

    return pd.DataFrame(model_combinations)


def get_combination(number):

    combinations_path = Path.cwd() / 'tools' / 'xlmroberta' / 'training_combinations.feather'
    if not combinations_path.exists():
        df = generate_combinations()
        df.to_feather(combinations_path)
    else:
        df = pd.read_feather(combinations_path)
    
    language, corpus = df.loc[number, ['languages', 'corpora']]

    return list(language), list(corpus)


def get_model_id_with_full_trainingdata():
    combinations_path = Path.cwd() / 'tools' / 'xlmroberta' / 'training_combinations.feather'
    if not combinations_path.exists():
        df = generate_combinations()
        df.to_feather(combinations_path)
    else:
        df = pd.read_feather(combinations_path)

    # exclude wikiann
    filt = df.corpora.apply(lambda x: 'wikiann' in x)
    df = df[~filt]

    filt = (df.languages.apply(len) == max(df.languages.apply(len))) & (df.corpora.apply(len) == max(df.corpora.apply(len)))
    
    return df[filt].index[0]


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
    results = _compute_metrics(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
        'raw_results': results
    }