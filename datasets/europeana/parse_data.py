from nltk.corpus.reader import ConllChunkCorpusReader
from pathlib import Path
import pandas as pd
cwd = Path.cwd()

p = cwd / 'datasets/' / 'europeana'
repository = p / 'ner-corpora'
tmp = p / 'tmp'

if not tmp.exists():
    tmp.mkdir()

for bio in repository.glob('*.bio/*.bio'):
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
        # manually patching some mistakes
        if l.startswith('Die Böhminger'):
            raw[i] = 'Die O\nBöhminger O\n'
        if l.startswith('„ . .'):
            raw[i] = "„ O\n. O\n. O\n"
        if l.startswith('. . .'):
            raw[i] = ". O\n. O\n. O\n"
        if l.startswith('. .'):
            raw[i] = ". O\n. O\n"
        if l.startswith(', (L3>'):
            raw[i] = ", O\n( O\nL O\n3 O\n> O\n"
        if l.startswith('L a'):
            raw[i] = "L O\na O\n"
        if l.startswith("zu haben"):
            raw[i] = "zu O\nhaben O\n"
        if l.startswith('Deutsche Bank'):
            raw[i] = "Deutsche B-ORG\nBank I-ORG\n"
        if l.startswith("S r ö"):
            raw[i] = "Srö O\n"
        if l.startswith("- -"):
            raw[i] = "- O\n- O\n"
        if l.startswith("— —"):
            raw[i] = "— O\n— O\n"
        if l.startswith("159.00— 100.50"):
            raw[i] = "159.00 O\n— O\n100.50 O\n"
    with open(tmp / bio.name.replace('.bio', '.txt'), 'w') as f:
        f.writelines(raw)


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