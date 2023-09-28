"""
    Utility functions to keep track on 
    downloaded corpora and tools
    keeping it simple by storing everything in a csv file
"""

from pathlib import Path
import csv
import pandas as pd

corpus_headers = ['corpus', 'subset', 'path', 'split', 'language', 'tokens', 'sentences']

def find_corpus_registry():
    p = Path.cwd()
    r = p / 'corpora' / 'registry.csv'
    if not r.exists():
        with open(r, 'w') as f:
            writer = csv.DictWriter(f, corpus_headers)
            writer.writeheader()
    return r

def add_corpus(data: dict):
    """ Add or update a corpus to the registry

        Corpora are uniquely identified by their absolute file path.
        This means that information for an already registered corpus
        are updated automatically as long as the absolute path matches.
    """
    assert isinstance(data, dict)
    h = corpus_headers.copy()
    assert list(data.keys()).sort() == h.sort()
    registry = find_corpus_registry()
    df = pd.read_csv(registry)
    data['path'] = str(data['path'])
    filt = df.path == data['path']
    df = df.loc[~filt]
    df = pd.concat([df, pd.DataFrame([data])], ignore_index=True)
    df.to_csv(registry, index=False)
    

def load_registry():
    r = find_corpus_registry()
    df = pd.read_csv(r)
    df = df.drop_duplicates()
    return df