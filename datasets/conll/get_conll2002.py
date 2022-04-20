# Script to automatically retrieve the CoNLL-2002 Dataset
# Introduction to the CoNLL-2002 Shared Task: Language-Independent Named Entity Recognition (Tjong Kim Sang, 2002)
# https://aclanthology.org/W02-2024/

import shutil
import requests
import tarfile
import gzip
from pathlib import Path
from nltk.corpus.reader import ConllChunkCorpusReader
import pandas as pd

url = "http://www.cnts.ua.ac.be/conll2002/ner.tgz"

cwd = Path.cwd()

p = cwd / 'datasets/' / 'conll'
tmp = p / 'tmp'


f_name =  p / 'conll2002.tgz'

if not f_name.exists():
    print(f'Downloading CoNLL-2002 Dataset from: {url}...')

    r = requests.get(url)

    with open(f_name, 'wb') as f:
        f.write(r.content)
    
    print(f'Success! Saved to: {f_name}')

print('Extracting archive...')
tar = tarfile.open(f_name)
tar.extractall(path=tmp)
tar.close()
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


for k, corp in esp_corps.items():
    ll = []
    for i, sent in enumerate(corp.tagged_sents(), start=1):
        for j, token in enumerate(sent, start=1):
            ll.append({'dataset': 'conll2002', 'language': 'es', 'corpus': k, 'sentence': i, 'token_id': j, 'token': token[0], 'IOB2': token[1]})
    df = pd.DataFrame(ll)
    df.to_feather(p / (k + '.feather'), compression='uncompressed')
    print(f"processed {k} and saved to {p / (k + '.feather')}")

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
            ll.append({'dataset': 'conll2002', 'language': 'nl', 'corpus': k.split('-')[0], 'doc': k.split('-')[1], 'sentence': i, 'token_id': j, 'token': token[0], 'POS': token[1], 'IOB2': token[2]})
    dfs.append(pd.DataFrame(ll))


df = pd.concat(dfs, ignore_index=True)

df.to_feather(p / 'ned.feather', compression='uncompressed')
print(f"Sucess! Saved to {p / 'ned.feather'}")

print(f'deleting temporary files...')
shutil.rmtree(tmp)
print(f'Done!')
