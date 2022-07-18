# Script to automatically retrieve Europeana NER Dataset
# An Open Corpus for Named Entity Recognition in Historic Newspapers (Neudecker, LREC 2016)
# https://aclanthology.org/L16-1689/
# https://github.com/EuropeanaNewspapers/ner-corpora

import sys
from pathlib import Path
import zipfile
import patch

from sklearn.model_selection import train_test_split

seed = 5618

sys.path.insert(0, str(Path.cwd()))
from utils.downloader import downloader
from utils.parsers import parse_conll
from utils.registry import add_corpus

p = Path.cwd() / 'corpora' / 'europeana'
tmp = p / 'tmp'
repository = tmp / 'ner-corpora-master'

if not tmp.exists():
    tmp.mkdir()

repo = "https://github.com/EuropeanaNewspapers/ner-corpora/archive/refs/heads/master.zip"

print(f'Downloading Europeana Dataset from: {repo}...')
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

for corpus in tmp.glob('*.txt'):
    df = parse_conll(corpus, columns=['token', 'CoNLL_IOB2'])
    df['dataset'] = 'europeana'
    df['subset'] = corpus.name.replace('.conll', '')
    language = ''
    if 'de' in corpus.name.lower():
        language = 'de'
    elif 'fr' in corpus.name.lower():
        language = 'fr'
    elif 'nl' in corpus.name.lower():
        language = 'nl'

    df['language'] = language
    df = df.drop(columns=['doc_id'])
    df.sentence_id = df.sentence_id.astype(str).str.zfill(7)
    # fix wrong tags
    df.CoNLL_IOB2 = df.CoNLL_IOB2.replace({'B-BER': 'B-PER'})
    df.CoNLL_IOB2 = df.CoNLL_IOB2.replace({'P': 'O'})

    sentence_index = df.sentence_id.unique().tolist()
    train, test_val = train_test_split(sentence_index, test_size=0.3, random_state=seed)
    test, val = train_test_split(test_val, test_size=0.5, random_state=seed)
    df_train = df.loc[df.sentence_id.isin(train), ]
    df_test = df.loc[df.sentence_id.isin(test), ]
    df_val = df.loc[df.sentence_id.isin(val), ]

    df_train.reset_index(inplace=True, drop=True)
    df_test.reset_index(inplace=True, drop=True)
    df_val.reset_index(inplace=True, drop=True)

    train_destination = p / corpus.name.replace('.txt', '_train.feather')
    test_destination = p / corpus.name.replace('.txt', '_test.feather')
    validation_destination = p / corpus.name.replace('.txt', '_validation.feather')

    df_train.to_feather(train_destination, compression='uncompressed')
    df_test.to_feather(test_destination, compression='uncompressed')
    df_val.to_feather(validation_destination, compression='uncompressed')
    print(f"processed {corpus} and saved to", train_destination, test_destination, validation_destination)

    train_details = {'corpus': 'europeana', 
                    'subset': corpus.name, 
                    'path': train_destination, 
                    'split': "train",
                    'language': language, 
                    'tokens': len(df_train), 
                    'sentences': len(df_train.sentence_id.unique())}

    add_corpus(train_details)

    test_details = {'corpus': 'europeana', 
                    'subset': corpus.name, 
                    'path': test_destination, 
                    'split': "test",
                    'language': language, 
                    'tokens': len(df_test), 
                    'sentences': len(df_test.sentence_id.unique())}

    add_corpus(test_details)

    validation_details = {'corpus': 'europeana', 
                    'subset': corpus.name, 
                    'path': validation_destination, 
                    'split': "validation",
                    'language': language, 
                    'tokens': len(df_val), 
                    'sentences': len(df_val.sentence_id.unique())}

    add_corpus(validation_details)


print('Done!')