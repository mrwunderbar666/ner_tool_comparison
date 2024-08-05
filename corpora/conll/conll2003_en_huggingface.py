import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd()))
# make sure to execute 'install_prerequisies.sh' before running this script
# it automatically fetches the Huggingface file which processes the Conll2003 english dataset
from utils.conll2003 import Conll2003
from utils.registry import add_corpus

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
    df['CoNLL_IOB2'] = df.ner_tags.replace(ner_tags)
    df['sentence_id'] = df.id.astype(int)
    df['sentence_id'] = df.sentence_id.astype(str).str.zfill(7)
    df['token_id'] = df.groupby('sentence_id').cumcount()
    df['token_id'] = df['token_id'] + 1
    df['token'] = df.tokens
    df['corpus'] = 'conll2003'
    df['language'] = 'en'
    df['subset'] = split

    df = df.loc[:, ['corpus', 'language', 'subset', 'sentence_id', 'token_id', 'token', 'CoNLL_IOB2']]
    df = df.loc[~df.token.isna(), ]
    df = df.reset_index(drop=True)
    corpus_destination = p / f'conll2003_en_{split}_iob.feather'
    df.to_feather(corpus_destination, compression = 'uncompressed')

    corpus_details = {'corpus': 'conll', 
                      'subset': split, 
                      'path': corpus_destination, 
                      'split': split,
                      'language': 'en', 
                      'tokens': len(df), 
                      'sentences': len(df.sentence_id.unique())}
    
    add_corpus(corpus_details)


