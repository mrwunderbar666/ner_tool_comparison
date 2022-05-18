# Script to automatically retrieve Emerging Entities Dataset
# Leon Derczynski, Eric Nichols, Marieke van Erp, Nut Limsopatham (2017) 
# "Results of the WNUT2017 Shared Task on Novel and Emerging Entity Recognition", 
# in Proceedings of the 3rd Workshop on Noisy, User-generated Text.
# https://noisy-text.github.io/2017/emerging-rare-entities.html

import sys
from pathlib import Path
import pandas as pd
from nltk.corpus.reader import ConllChunkCorpusReader

sys.path.append(str(Path.cwd()))
from utils.downloader import downloader

p = Path.cwd() / 'corpora' / 'emerging'
tmp = p / 'tmp'

if not tmp.exists():
    tmp.mkdir()


training_url = "https://noisy-text.github.io/2017/files/wnut17train.conll"
dev_url = "https://noisy-text.github.io/2017/files/emerging.dev.conll"
test_url = "https://noisy-text.github.io/2017/files/emerging.test"
test_with_tags = "https://noisy-text.github.io/2017/files/emerging.test.annotated"

print(f'Downloading Emerging Entities Training Data from: {training_url}...')
downloader(training_url, tmp / 'wnut17train.conll')

print(f'Downloading Emerging Entities Dev Data from: {dev_url}...')
downloader(dev_url, tmp / 'emerging.dev.conll')

with open(tmp / 'emerging.dev.conll', 'r') as f_in:
    l = f_in.readlines()

with open(tmp / 'emerging.dev.conll', 'w') as f_out: 
    f_out.writelines(l[:-1])

print(f'Downloading Emerging Entities Test Data from: {test_with_tags}...')
downloader(test_with_tags, tmp / 'emerging.test.annotated.conll')

corps = {}

for txt in tmp.glob('*.conll'):
    corp = ConllChunkCorpusReader(f'{tmp}', [txt.name], ['words', 'ne'], separator='\t', encoding='utf-8-sig')
    corps[txt.name.replace('.conll', '')] = corp

print('Processing...')

# map to conll iob2 format
emerging2conll = {'person': 'PER', 
                    'creative-work': 'MISC', 
                    'group': 'ORG', 
                    'location': 'LOC', 
                    'product': 'MISC', 
                    'corporation': 'ORG'}

for k, corp in corps.items():
    ll = []
    for i, sent in enumerate(corp.tagged_sents(), start=1):
        for j, token in enumerate(sent, start=1):
            ll.append({'dataset': 'emerging_entities', 'language': 'en', 'corpus': k, 'sentence_id': i, 'token_id': j, 'token': token[0], 'IOB2': token[1]})
    df = pd.DataFrame(ll)
    df['CoNLL_IOB2'] = df.IOB2.replace(emerging2conll, regex=True)
    df.to_feather(p / (k + '.feather'), compression='uncompressed')
    print(f"processed {k} and saved to {p / (k + '.feather')}")

print('Done!')