# Script to automatically retrieve Europeana NER Dataset
# An Open Corpus for Named Entity Recognition in Historic Newspapers (Neudecker, LREC 2016)
# https://aclanthology.org/L16-1689/
# https://github.com/EuropeanaNewspapers/ner-corpora

import requests
from pathlib import Path
import pandas as pd
import zipfile
import io
import patch
from nltk.corpus.reader import ConllChunkCorpusReader
import shutil

p = Path.cwd() / 'datasets' / 'europeana'
tmp = p / 'tmp'
repository = p / 'ner-corpora-master'

if not tmp.exists():
    tmp.mkdir()

repo = "https://github.com/EuropeanaNewspapers/ner-corpora/archive/refs/heads/master.zip"

print(f'Downloading Europeana Dataset from: {repo}...')
r = requests.get(repo)
print('Success!')

print('Extracting archive...')
z = zipfile.ZipFile(io.BytesIO(r.content), mode='r')
z.extractall(path=p)
print('ok')

# Apply patch
# Provided dataset does not comply with CoNLL specifications
# To reduce the noise, some patching is necessary
print('Applying fixes to "enp_DE.sbb"...')
fixes = patch.fromfile(p / 'enp_DE.sbb.patch')
fixes.apply()
print('ok')

print('Preprocessing...')
for bio in repository.glob('*.bio/*.bio'):
    print(f'{bio}')
    with open(bio, 'r') as f:
        raw = f.readlines()
    for i, l in enumerate(raw):
        # brute force separate sentences
        if l.startswith('. O'):
            raw[i] += '\n'
        # remove comments
        elif l.startswith('# '):
            raw[i] = '\n'
        # tidy up missing annotations
        if ' ' not in l:
            raw[i] = l.replace('\n', ' O\n')
    with open(tmp / bio.name.replace('.bio', '.txt'), 'w') as f:
        f.writelines(raw)

print('ok')

corps = {}

for txt in tmp.glob('*.txt'):
    corp = ConllChunkCorpusReader(f'{tmp}', [txt.name], ['words', 'ne'])
    corps[txt.name.replace('.txt', '')] = corp


for k, corp in corps.items():
    ll = []
    for i, sent in enumerate(corp.tagged_sents(), start=1):
        for j, token in enumerate(sent, start=1):
            ll.append({'dataset': 'europeana', 'language': k.split('_')[-1].split('.')[0].lower(), 'corpus': k, 'sentence': i, 'token_id': j, 'token': token[0], 'IOB2': token[1]})
    df = pd.DataFrame(ll)
    df.to_feather(p / (k + '.feather'), compression='uncompressed')
    print(f"processed {k} and saved to {p / (k + '.feather')}")


shutil.rmtree(tmp)
shutil.rmtree(p / 'ner-corpora-master')
print('Done!')