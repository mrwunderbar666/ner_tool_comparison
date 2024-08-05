# Script to automatically retrieve Emerging Entities Dataset
# Leon Derczynski, Eric Nichols, Marieke van Erp, Nut Limsopatham (2017) 
# "Results of the WNUT2017 Shared Task on Novel and Emerging Entity Recognition", 
# in Proceedings of the 3rd Workshop on Noisy, User-generated Text.
# https://noisy-text.github.io/2017/emerging-rare-entities.html

import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from utils.downloader import downloader
from utils.parsers import parse_conll
from utils.registry import add_corpus
from utils.mappings import emerging2conll

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

print(f'Downloading Emerging Entities Test Data from: {test_with_tags}...')
downloader(test_with_tags, tmp / 'emerging.test.annotated.conll')

# map to conll iob2 format

print('Processing...')
for corpus in tmp.glob('*.conll'):
    df = parse_conll(corpus, columns=['token', 'IOB2'], encoding='utf-8-sig', separator='\t')
    df['corpus'] = 'emerging'
    df['subset'] = corpus.name.replace('.conll', '')
    df['language'] = 'en'
    df = df.drop(columns=['doc_id'])
    df.sentence_id = df.sentence_id.astype(str).str.zfill(7)
    df['CoNLL_IOB2'] = df.IOB2.replace(emerging2conll, regex=True)
    corpus_destination = p / corpus.name.replace('.conll', '.feather')
    df.to_feather(corpus_destination, compression='uncompressed')
    print(f"processed {corpus} and saved to {corpus_destination}")

    split = ''
    if "test" in corpus.name:
        split = 'validation'
    elif "dev" in corpus.name:
        split = 'test'
    elif "train" in corpus.name:
        split = "train"

    corpus_details = {'corpus': 'emerging', 
                    'subset': corpus.name, 
                    'path': corpus_destination, 
                    'split': split,
                    'language': 'en', 
                    'tokens': len(df), 
                    'sentences': len(df.sentence_id.unique())}

    add_corpus(corpus_details)

print('Done!')