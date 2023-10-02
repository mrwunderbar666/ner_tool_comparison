import sys
import zipfile

from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path.cwd()))
from utils.registry import add_corpus
from utils.parsers import parse_conll

p = Path.cwd() / 'corpora' / 'kind'

if __name__ == '__main__':

    tmp = p / 'tmp'

    if not tmp.exists():
        tmp.mkdir(parents=True)

    z = zipfile.ZipFile(p / "evalita_2023.zip", mode='r')
    z.extractall(path=tmp)

    docs = []

    for corpus in tmp.glob("*.tsv"):
        print('Parsing file', corpus.name)

        df = parse_conll(corpus, columns=['token', 'CoNLL_IOB2'], separator="\t")

        df['sentence_id'] = df['sentence_id'].astype(str).str.zfill(7)
        df['language'] = 'it'

        df['corpus'] = 'kind'
        df['subset'] = corpus.name.split('_')[0]

        cols = ['corpus', 'subset',
                'language', 'sentence_id', 
                'token_id', 
                'token', 'CoNLL_IOB2']

        df = df.loc[:, cols]
        corpus_destination = p / corpus.name.replace('.tsv', '.feather')
        df.to_feather(corpus_destination, compression='uncompressed')

        split = ''
        if "test" in corpus.name:
            split = 'validation'
        elif "dev" in corpus.name:
            split = 'test'
        elif "train" in corpus.name:
            split = "train"

        corpus_details = {'corpus': 'kind', 
                            'subset': corpus.name.split('_')[0], 
                            'path': corpus_destination, 
                            'split': split,
                            'language': 'it', 
                            'tokens': len(df), 
                            'sentences': len(df.sentence_id.unique())}

        add_corpus(corpus_details)
        print(f"Sucess! Saved to {corpus_destination}")

print('Done!')