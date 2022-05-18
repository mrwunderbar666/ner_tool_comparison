# Script to automatically retrieve Europeana NER Dataset
# An Open Corpus for Named Entity Recognition in Historic Newspapers (Neudecker, LREC 2016)
# https://aclanthology.org/L16-1689/
# https://github.com/EuropeanaNewspapers/ner-corpora

import sys
import requests
from pathlib import Path
import pandas as pd
import zipfile
import io
import patch
from nltk.corpus.reader import ConllChunkCorpusReader

from sklearn.model_selection import train_test_split

seed = 5618

sys.path.append(str(Path.cwd()))
from utils.downloader import downloader

p = Path.cwd() / 'corpora' / 'europeana'
tmp = p / 'tmp'
repository = tmp / 'ner-corpora-master'

if not tmp.exists():
    tmp.mkdir()

repo = "https://github.com/EuropeanaNewspapers/ner-corpora/archive/refs/heads/master.zip"

print(f'Downloading Europeana Dataset from: {repo}...')
r = requests.get(repo)
downloader(repo, tmp / 'master.zip')

print('Extracting archive...')
z = zipfile.ZipFile(tmp / 'master.zip', mode='r')
z.extractall(path=tmp)
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
    print('Parsing', k, 'to dataframe')
    ll = []
    for i, sent in enumerate(corp.tagged_sents(), start=1):
        for j, token in enumerate(sent, start=1):
            ll.append({'dataset': 'europeana', 'language': k.split('_')[-1].split('.')[0].lower(), 'corpus': k, 'sentence_id': i, 'token_id': j, 'token': token[0], 'IOB2': token[1]})

    df = pd.DataFrame(ll)
    # fixing wrong tags
    df.IOB2 = df.IOB2.replace({'B-BER': 'B-PER'})
    df.IOB2 = df.IOB2.replace({'P': 'O'})
    sentence_index = df.sentence_id.unique().tolist()
    train, test_val = train_test_split(sentence_index, test_size=0.3, random_state=seed)
    test, val = train_test_split(test_val, test_size=0.5, random_state=seed)
    df_train = df.loc[df.sentence_id.isin(train), ]
    df_test = df.loc[df.sentence_id.isin(test), ]
    df_val = df.loc[df.sentence_id.isin(val), ]

    df_train.reset_index(inplace=True, drop=True)
    df_test.reset_index(inplace=True, drop=True)
    df_val.reset_index(inplace=True, drop=True)

    df_train.to_feather(p / (k + '_train.feather'), compression='uncompressed')
    df_test.to_feather(p / (k + '_test.feather'), compression='uncompressed')
    df_val.to_feather(p / (k + '_validation.feather'), compression='uncompressed')
    print(f"processed {k} and saved to {p / (k + '.feather')}")


print('Done!')