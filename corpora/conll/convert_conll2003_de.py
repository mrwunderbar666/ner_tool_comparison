import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path.cwd()))
from utils.registry import add_corpus
from utils.parsers import parse_conll

p = Path.cwd() / 'corpora/' / 'conll'

raw_data = [p / 'deu.testa', p / 'deu.testb', p / 'deu.train']

for corpus in raw_data:
    assert corpus.exists(), f"Could not find raw corpus data: {corpus}"
    print('Converting', corpus)
    df = parse_conll(corpus, encoding='latin-1')

    """ 
        CoNLL-2003 uses a legacy format, the documentation reads:

        The chunk tags and the named entity tags have the format I-TYPE 
        which means that the word is inside a phrase of type TYPE. 
        __Only if two phrases of the same type immediately follow each other__, 
        the first word of the second phrase will have tag B-TYPE 
        to show that it starts a new phrase. 

        This means that all tags are by default I-TAGs, which is different from
        the other corpora (where the Beginning is always marked with a B-TAG).
        Here we fix this with a simple reformatting
    """

    df['i_tag'] = df.CoNLL_IOB2.str.startswith('I')
    df['previous_i_tag'] = df.CoNLL_IOB2.shift(1).str.startswith('I').fillna(False)

    filt = (df.i_tag) & (~df.previous_i_tag)

    df.loc[filt, 'CoNLL_IOB2'] = df.loc[filt, 'CoNLL_IOB2'].str.replace('I-', 'B-', regex=False)

    df['corpus'] = 'conll2003'
    df['subset'] = corpus.name
    df['language'] = 'de'
    df.sentence_id = df.sentence_id.astype(str).str.zfill(7)
    df.doc_id = df.doc_id.astype(str).str.zfill(7)
    df.sentence_id = df.doc_id + '_' + df.sentence_id
    corpus_destination = str(corpus) + '.feather'
    df.to_feather(corpus_destination, compression='uncompressed')

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

print('Done!')