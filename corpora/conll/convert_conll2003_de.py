import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path.cwd()))
from utils.registry import add_corpus
from utils.parsers import parse_conll

p = Path.cwd() / 'corpora/' / 'conll'

raw_data = [p / 'deu.testa', p / 'deu.testb', p / 'deu.train']

for corpus in raw_data:
    assert corpus.exists()
    print('Converting', corpus)
    df = parse_conll(corpus, encoding='latin-1')
    df['dataset'] = 'conll2003'
    df['subset'] = corpus.name
    df['language'] = 'de'
    df.sentence_id = df.sentence_id.astype(str).str.zfill(6)
    df.doc_id = df.doc_id.astype(str).str.zfill(4)
    df.sentence_id = df.doc_id + '_' + df.sentence_id
    corpus_destination = str(corpus) + '.feather'
    df.to_feather(corpus_destination)

    split = ''
    if "testb" in corpus.name:
        split = 'validation'
    elif "testa" in corpus.name:
        split = 'test'
    elif "train" in corpus.name:
        split = "train"
    
    corpus_details = {'corpus': 'conll', 
                      'subset': corpus.name, 
                      'path': corpus_destination, 
                      'split': split,
                      'language': 'de', 
                      'tokens': len(df), 
                      'sentences': sum(df.groupby('doc_id').sentence_id.unique().apply(lambda x: len(x)))}
    
    add_corpus(corpus_details)