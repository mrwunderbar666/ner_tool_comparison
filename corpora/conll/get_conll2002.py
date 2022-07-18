# Script to automatically retrieve the CoNLL-2002 Dataset
# Introduction to the CoNLL-2002 Shared Task: Language-Independent Named Entity Recognition (Tjong Kim Sang, 2002)
# https://aclanthology.org/W02-2024/

import sys
import os
import tarfile
import gzip
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path.cwd()))
from utils.downloader import downloader
from utils.parsers import parse_conll
from utils.registry import add_corpus

url = "http://www.cnts.ua.ac.be/conll2002/ner.tgz"

cwd = Path.cwd()

p = cwd / 'corpora/' / 'conll'
tmp = p / 'tmp'

if not tmp.exists():
    tmp.mkdir()

f_name =  tmp / 'conll2002.tgz'

print(f'Downloading CoNLL-2002 Dataset from: {url}...')
downloader(url, f_name)

print('Extracting archive...')
with tarfile.open(f_name, 'r:gz', errorlevel=1) as tar:
    for f in tar:
        try:
            tar.extract(f, path=tmp)
        except IOError as e:
            os.remove(tmp / f.name)
            tar.extract(f, path=tmp)
        finally:
            os.chmod(tmp / f.name, f.mode)

print('ok')

print('Deflating data...')
for gz in tmp.glob('ner/data/*.gz'):
    with open(gz, 'rb') as f_in:
        decompressed = gzip.decompress(f_in.read())
    with open(tmp / gz.name.replace('.gz', '.txt'), 'wb') as f_out:
        f_out.write(decompressed)

print('ok')


print('Processing Spanish corpus...')

spanish_corpora = [tmp / 'esp.testa.txt', tmp / 'esp.testb.txt', tmp / 'esp.train.txt']

for corpus in spanish_corpora:
    assert corpus.exists()
    print('Converting', corpus)
    df = parse_conll(corpus, columns=['token', 'CoNLL_IOB2'], encoding='latin-1')
    df['dataset'] = 'conll2003'
    df['subset'] = corpus.name
    df['language'] = 'es'
    df = df.drop(columns=['doc_id'])
    df.sentence_id = df.sentence_id.astype(str).str.zfill(7)
    corpus_destination = p / corpus.name.replace('.txt', '.feather')
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
                      'language': 'es', 
                      'tokens': len(df), 
                      'sentences': len(df.sentence_id.unique())}
    
    add_corpus(corpus_details)
    print(f"Sucess! Saved to {corpus_destination}")

print('Processing Dutch corpus...')

dutch_corpora = [tmp / 'ned.testa.txt', tmp / 'ned.testb.txt', tmp / 'ned.train.txt']

for corpus in dutch_corpora:
    assert corpus.exists()
    print('Converting', corpus)
    df = parse_conll(corpus, columns=['token', 'POS', 'CoNLL_IOB2'], encoding='latin-1')
    df['dataset'] = 'conll2003'
    df['subset'] = corpus.name
    df['language'] = 'nl'
    df.sentence_id = df.sentence_id.astype(str).str.zfill(7)
    df.doc_id = df.doc_id.astype(str).str.zfill(7)
    df.sentence_id = df.doc_id + '_' + df.sentence_id

    corpus_destination = p / corpus.name.replace('.txt', '.feather')
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
                      'language': 'nl', 
                      'tokens': len(df), 
                      'sentences': sum(df.groupby('doc_id').sentence_id.unique().apply(lambda x: len(x)))}
    
    add_corpus(corpus_details)
    print(f"Sucess! Saved to {corpus_destination}")

print(f'Done!')
