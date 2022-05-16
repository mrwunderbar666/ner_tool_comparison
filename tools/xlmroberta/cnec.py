import pandas as pd
from pathlib import Path
from datasets import ClassLabel, Features, Dataset, Value, Sequence


# Load and prepare data

p_cncec = Path.cwd() / 'corpora' / 'cnec'
df_train = pd.read_feather(p_cncec / 'named_ent_train.feather')
df_test = pd.read_feather(p_cncec / 'named_ent_dtest.feather')
df_validation = pd.read_feather(p_cncec / 'named_ent_etest.feather')


labels_dict = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
labels = ClassLabel(num_classes=len(labels_dict.keys()), names=list(labels_dict.keys()))
features = Features({'text': Sequence(Value(dtype="string")), 'labels': Sequence(labels)})

df_train['CoNLL_IOB2'] = df_train['CoNLL_IOB2'].replace(labels_dict)

df_train = df_train.groupby('sentence_id')[['token', 'CoNLL_IOB2']].agg(list)
df_train = df_train.rename(columns={'token': 'text', 'CoNLL_IOB2': 'labels'})

dataset_train = Dataset.from_pandas(df_train, features=features)

df_test['CoNLL_IOB2'] = df_test['CoNLL_IOB2'].replace(labels_dict)

df_test = df_test.groupby('sentence_id')[['token', 'CoNLL_IOB2']].agg(list)
df_test = df_test.rename(columns={'token': 'text', 'CoNLL_IOB2': 'labels'})

dataset_test = Dataset.from_pandas(df_test, features=features)

df_validation['CoNLL_IOB2'] = df_validation['CoNLL_IOB2'].replace(labels_dict)

df_validation = df_validation.groupby('sentence_id')[['token', 'CoNLL_IOB2']].agg(list)
df_validation = df_validation.rename(columns={'token': 'text', 'CoNLL_IOB2': 'labels'})

dataset_validation = Dataset.from_pandas(df_validation, features=features)

# Tokenize

# Set-up training parameters

# Train

# Evaluate

# Save results