import sys
from pathlib import Path
sys.path.append(str(Path.cwd()))
from utils.conll2003 import Conll2003

p = Path.cwd() / 'corpora' / 'conll' 

ner_tags = {0: "O",
            1: "B-PER",
            2: "I-PER",
            3: "B-ORG",
            4: "I-ORG",
            5: "B-LOC",
            6: "I-LOC",
            7: "B-MISC",
            8: "I-MISC"}

Conll2003().download_and_prepare()
conll = Conll2003().as_dataset()

for split in conll.keys():

    df = conll[split]

    df = df.to_pandas()
    df.to_feather(p / f'conll2003_en_{split}_raw.feather', compression = 'uncompressed')

    df = df.explode(['tokens', 'pos_tags', 'chunk_tags', 'ner_tags'])
    df['IOB2'] = df.ner_tags.replace(ner_tags)
    df['sentence_id'] = df.id
    df['token_id'] = df.groupby('sentence_id').cumcount()
    df['token_id'] = df['token_id'] + 1
    df['token'] = df.tokens
    df['dataset'] = 'conll2003'
    df['language'] = 'en'
    df['corpus'] = split

    df = df.loc[:, ['dataset', 'language', 'corpus', 'sentence_id', 'token_id', 'token', 'IOB2']]

    df = df.reset_index(drop=True)

    df.to_feather(p / f'conll2003_en_{split}_iob.feather', compression = 'uncompressed')


