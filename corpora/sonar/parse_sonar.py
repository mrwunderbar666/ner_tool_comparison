import sys

from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path.cwd()))
from utils.registry import add_corpus
from utils.parsers import parse_conll
from utils.mappings import sonar2conll

p = Path.cwd() / 'corpora' / 'sonar'

seed = 5618

text_sources = {
    "WR-P-P-B": "Books",  
    "WR-P-P-C": "Brochures",
    "WR-P-E-C": "E-magazines",
    "WR-P-E-E": "E-Newsletters",
    "WR-P-P-E": "Guides, Manuals",
    "WR-P-P-G": "Newspapers",
    "WR-P-P-F": "Legal texts",
    "WR-P-P-I": "Magazines",
    "WR-P-P-E": "Newsletters",
    "WR-P-P-J": "Policy documents",
    "WR-P-P-H": "Newspapers",
    "WS-U-E-A": "Autocues",
    "WR-U-T-A": "Minutes",
    "WR-P-E-J": "Press releases",
    "WR-P-P-K": "Proceedings",
    "WR-P-P-L": "Reports",
    "WS-U-E-B": "Speeches",
    "WR-P-E-H": "Teletext",
    "WR-P-E-I": "Websites", # not sure about this one
    "WR-P-E-J": "Wikipedia",
    "wiki": "Wikipedia",
    "dpc": "Dutch Parallel Corpus"
}

if __name__ == '__main__':

    tmp = p / 'tmp'

    files = list(tmp.glob("IOB/*.iob"))

    assert len(files) > 0, 'Could not find iob files in tmp dir. Please retrieve the corpus and drop the files in tmp/'

    docs = []

    for doc in tmp.glob("IOB/*.iob"):
        print('Parsing file', doc.name)

        df = parse_conll(doc, columns=['token', 'ner_tag'], separator="\t")
        df['doc_id'] = doc.name.replace('.iob', 'iob')

        if doc.name.startswith('wiki'):
            subset = "wikipedia"
        elif doc.name.startswith('dpc'):
            subset = "Dutch Parallel Corpus"
        else:
            subset = text_sources["-".join(doc.name.split('-')[:-1])]
        
        df['subset'] = subset
        docs.append(df)

    corpus = pd.concat(docs, ignore_index=True).reset_index(drop=True)

    corpus['sentence_id'] = corpus['doc_id'] + '_' + corpus['sentence_id'].astype(str).str.zfill(4)

    corpus['CoNLL_IOB2'] = corpus.ner_tag.str.upper().replace(sonar2conll)

    corpus['language'] = 'nl'

    corpus['corpus'] = 'sonar'

    cols = ['corpus', 'subset',
            'language', 'sentence_id', 
            'token_id', 
            'token', 'CoNLL_IOB2', 'ner_tag']

    corpus = corpus.loc[:, cols]

    sentence_index = corpus.sentence_id.unique().tolist()
    train, test_val = train_test_split(sentence_index, test_size=0.3, random_state=seed)
    test, val = train_test_split(test_val, test_size=0.5, random_state=seed)

    corpus_train = corpus.loc[corpus.sentence_id.isin(train), :]
    corpus_test = corpus.loc[corpus.sentence_id.isin(test), :]
    corpus_val = corpus.loc[corpus.sentence_id.isin(val), :]

    corpus_train.reset_index(inplace=True, drop=True)
    corpus_test.reset_index(inplace=True, drop=True)
    corpus_val.reset_index(inplace=True, drop=True)

    train_destination = p / 'sonar_train.feather'
    test_destination = p / 'sonar_test.feather'
    validation_destination = p / 'sonar_validation.feather'

    corpus_train.to_feather(train_destination, compression='uncompressed')
    corpus_test.to_feather(test_destination, compression='uncompressed')
    corpus_val.to_feather(validation_destination, compression='uncompressed')
    print(f"processed SoNaR-1 and saved to", train_destination, test_destination, validation_destination)

    train_details = {'corpus': 'sonar', 
                        'subset': 'SoNaR-1', 
                        'path': train_destination, 
                        'split': 'train',
                        'language': 'nl', 
                        'tokens': len(corpus_train), 
                        'sentences': len(corpus_train.sentence_id.unique())}

    add_corpus(train_details)

    test_details = {'corpus': 'sonar', 
                        'subset': 'SoNaR-1', 
                        'path': test_destination, 
                        'split': 'test',
                        'language': 'nl', 
                        'tokens': len(corpus_test), 
                        'sentences': len(corpus_test.sentence_id.unique())}

    add_corpus(test_details)

    validation_details = {'corpus': 'sonar', 
                        'subset': 'SoNaR-1', 
                        'path': validation_destination, 
                        'split': 'validation',
                        'language': 'nl', 
                        'tokens': len(corpus_val), 
                        'sentences': len(corpus_val.sentence_id.unique())}

    add_corpus(validation_details)



print('Done!')