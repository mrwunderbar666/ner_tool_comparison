# Script to automatically retrieve the CoNLL-2002 Dataset
# Introduction to the CoNLL-2002 Shared Task: Language-Independent Named Entity Recognition (Tjong Kim Sang, 2002)
# https://aclanthology.org/W02-2024/

import sys
import os
import tarfile
import gzip
from pathlib import Path
from nltk.corpus.reader import ConllChunkCorpusReader
from nltk.tokenize.treebank import TreebankWordDetokenizer
import pandas as pd

sys.path.insert(0, str(Path.cwd()))
from utils.downloader import downloader
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
esp_corps = {}

for txt in tmp.glob('esp*.txt'):
    corp = ConllChunkCorpusReader(f'{tmp}', [txt.name], ['words', 'ne'], encoding='latin-1')
    esp_corps[txt.name.replace('.txt', '')] = corp

twd = TreebankWordDetokenizer()

for k, corp in esp_corps.items():
    ll = []
    for i, sent in enumerate(corp.tagged_sents(), start=1):
        for j, token in enumerate(sent, start=1):
            ll.append({'dataset': 'conll2002', 'language': 'es', 'corpus': k, 'sentence_id': i, 'token_id': j, 'token': token[0], 'CoNLL_IOB2': token[1]})
    df = pd.DataFrame(ll)
    corpus_destination = p / (k + '.feather')
    df.to_feather(corpus_destination, compression='uncompressed')
    print(f"processed {k} and saved to {corpus_destination}")
    sentences = df.groupby('sentence_id').token.apply(lambda x: twd.detokenize(x))
    with open(p / (k + '.txt'), 'w') as txt:
        txt.write("\n".join(sentences.to_list()))
    
    split = ''
    if "testb" in k:
        split = 'validation'
    elif "testa" in k:
        split = 'test'
    elif "train" in k:
        split = "train"
    
    corpus_details = {'corpus': 'conll', 
                      'subset': k, 
                      'path': corpus_destination, 
                      'split': split,
                      'language': 'es', 
                      'tokens': len(df), 
                      'sentences': len(sentences)}
    
    add_corpus(corpus_details)

print('Processing Dutch corpus...')
# split dutch corp files into documents
for txt in tmp.glob('ned*.txt'):
    Path.mkdir(tmp / txt.name.replace('.txt', ''), exist_ok=True)
    with open(txt, 'rb') as f:
        corp = f.read().decode('latin-1')
        corp = corp.split('-DOCSTART- -DOCSTART- O')
        for i, doc in enumerate(corp):
            if len(doc) < 2:
                continue
            with open(tmp / txt.name.replace('.txt', '') / f'doc_{i:03}.txt', 'w') as f_out:
                f_out.write(doc) 



dutch_corps = {}

for txt in tmp.glob('ned*/*.txt'):
    corp = ConllChunkCorpusReader(f'{txt.parent}', [txt.name], ['words', 'pos', 'ne'])
    dutch_corps[txt.parts[-2] + '-' + txt.name.replace('.txt', '')] = corp

dfs = []

for k, corp in dutch_corps.items():
    ll = []
    for i, sent in enumerate(corp.iob_sents(), start=1):
        for j, token in enumerate(sent, start=1):
            ll.append({'dataset': 'conll2002', 'language': 'nl', 'corpus': k.split('-')[0], 'doc': k.split('-')[1], 'sentence_id': i, 'token_id': j, 'token': token[0], 'POS': token[1], 'CoNLL_IOB2': token[2]})
    tmp_df = pd.DataFrame(ll)
    sentences = tmp_df.groupby('sentence_id').token.apply(lambda x: twd.detokenize(x))
    with open(p / (k.split('-')[0] + '.txt'), 'a+') as txt:
        txt.write("\n".join(sentences.to_list()))
    dfs.append(pd.DataFrame(ll))

df = pd.concat(dfs, ignore_index=True)

for corpus in df.corpus.unique():
    
    tmp_df = df.loc[df.corpus == corpus, :]
    tmp_df = tmp_df.reset_index(drop=True)

    corpus_destination = p / f"{corpus}.feather"
    tmp_df.to_feather(corpus_destination, compression='uncompressed')


    split = ''
    if "testb" in corpus:
        split = 'validation'
    elif "testa" in corpus:
        split = 'test'
    elif "train" in corpus:
        split = "train"
    
    corpus_details = {'corpus': 'conll', 
                      'subset': corpus, 
                      'path': corpus_destination, 
                      'split': split,
                      'language': 'nl', 
                      'tokens': len(tmp_df), 
                      'sentences': len(tmp_df.sentence_id.unique())}
    
    add_corpus(corpus_details)

    print(f"Sucess! Saved to {f_name}")

print(f'Done!')
