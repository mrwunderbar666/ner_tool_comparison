import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))
from utils.registry import add_corpus
from utils.downloader import downloader
from utils.parsers import parse_conll
from utils.mappings import finer2conll

dev_url = "https://github.com/mpsilfve/finer-data/raw/master/data/digitoday.2014.dev.csv"
train_url = "https://github.com/mpsilfve/finer-data/raw/master/data/digitoday.2014.train.csv"
test_url = "https://github.com/mpsilfve/finer-data/raw/master/data/digitoday.2015.test.csv"


p = Path.cwd() / 'corpora' / 'finer'

if __name__ == '__main__':

    tmp = p / 'tmp'

    if not tmp.exists():
        tmp.mkdir(parents=True)

    print('downloading FiNER files ...')
    downloader(dev_url, tmp / 'digitoday.2014.dev.txt')
    downloader(train_url, tmp / 'digitoday.2014.train.txt')
    downloader(test_url, tmp / 'digitoday.2015.test.txt')

    files = [tmp / 'digitoday.2014.dev.txt', tmp / 'digitoday.2014.train.txt', tmp / 'digitoday.2015.test.txt']

    for corpus in files:

        df = parse_conll(corpus, 
                        columns=['token', 'ner_tag', 'nested_ner_tag'],
                        separator='\t')

        df['CoNLL_IOB2'] = df.ner_tag.replace(finer2conll)
        df['corpus'] = 'finer'
        df['subset'] = corpus.name.replace('.txt', '')
        df['language'] = 'fi'
        df.sentence_id = df.doc_id.astype(str).str.zfill(4) + '_' + df.sentence_id.astype(str).str.zfill(4)
        corpus_destination = p / corpus.name.replace('.txt', '.feather')
        df.to_feather(corpus_destination, compression='uncompressed')

        split = ''
        if "test" in corpus.name:
            split = 'validation'
        elif "dev" in corpus.name:
            split = 'test'
        elif "train" in corpus.name:
            split = "train"

        corpus_details = {'corpus': 'finer', 
                            'subset': corpus.name.replace('.txt', ''), 
                            'path': corpus_destination, 
                            'split': split,
                            'language': 'fi', 
                            'tokens': len(df), 
                            'sentences': len(df.sentence_id.unique())}

        add_corpus(corpus_details)
        print(f"Sucess! Saved to {corpus_destination}")

    print('Done!')