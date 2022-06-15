# Utility functions to keep track on 
# downloaded corpora and tools
# keeping it simple by storing everything in a csv file

from pathlib import Path
import csv

corpus_headers = ['corpus', 'subset', 'path', 'split', 'language', 'tokens', 'sentences']

def find_corpus_registry():
    p = Path.cwd()
    r = p / 'corpora' / 'registry.csv'
    if not r.exists():
        with open(r, 'w') as f:
            writer = csv.DictWriter(f, corpus_headers)
            writer.writeheader()
    return r

def add_corpus(data):
    assert isinstance(data, dict)
    h = corpus_headers.copy()
    assert list(data.keys()).sort() == h.sort()
    registry = find_corpus_registry()
    with open(registry, 'a+') as f:
        writer = csv.DictWriter(f, corpus_headers)
        writer.writerow(data)
