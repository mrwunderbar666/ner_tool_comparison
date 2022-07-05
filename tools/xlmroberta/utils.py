import itertools
from pathlib import Path
from transformers import AutoTokenizer, DataCollatorForTokenClassification
from datasets import ClassLabel, Features, Value, Sequence
from utils.registry import load_registry
import pandas as pd

labels_dict = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}

conll_labels = ClassLabel(num_classes=len(labels_dict.keys()), names=list(labels_dict.keys()))
conll_features = Features({'text': Sequence(Value(dtype="string")), 'labels': Sequence(conll_labels)})

tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

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


def generate_combinations():

    language_combinations = [
        ['en'], ['de'], ['es'], ['nl'], ['fr'], ['zh'], ['ar'], ['cs'], # every language individually
        ['en', 'de', 'es', 'nl', 'fr', 'zh', 'ar', 'cs'], # all languages together
        ['en', 'de', 'es', 'nl', 'fr', 'cs'] # only using latin scripts
    ]

    registry = load_registry()
    registry = registry.loc[~(registry.corpus == 'wikiann')]

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
    
    language, corpus = df.loc[number]

    return list(language), list(corpus)