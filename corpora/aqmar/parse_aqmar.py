import sys
import copy
import typing as t
import zipfile

from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from bs4 import BeautifulSoup
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd

sys.path.insert(0, str(Path.cwd()))
from utils.registry import add_corpus
from utils.parsers import parse_conll

p = Path.cwd() / 'corpora' / 'aqmar'

seed = 5618

if __name__ == '__main__':

    tmp = p / 'tmp'

    if not tmp.exists():
        tmp.mkdir(parents=True)

    z = zipfile.ZipFile(p / "AQMAR_Arabic_NER_corpus-1.0.zip", mode='r')
    z.extractall(path=tmp)

    docs = []

    for txt in tmp.glob('*.txt'):
        df = parse_conll(txt, columns=['token', 'ner_tag'])
        df['doc_id'] = txt.name.replace('.txt', '')
        # map Misc tags to generic misc tag
        docs.append(df)

    df = pd.concat(docs, ignore_index=True).reset_index(drop=True)

    df['sentence_id'] = df['doc_id'] + '_' + df['sentence_id'].astype(str).str.zfill(4)
    df['CoNLL_IOB2'] = df.ner_tag.str.replace(r'([IB]-)MIS.*', r'\1MISC', regex=True)

    # fix some wrong labels
    df['CoNLL_IOB2'] = df.CoNLL_IOB2.str.replace('OO', 'O')
    df['CoNLL_IOB2'] = df.CoNLL_IOB2.str.replace('IO', 'O')
    df['CoNLL_IOB2'] = df.CoNLL_IOB2.str.replace('B-SPANISH', 'B-MISC')
    df['CoNLL_IOB2'] = df.CoNLL_IOB2.str.replace('B-ENGLISH', 'B-MISC')
    df['CoNLL_IOB2'] = df.CoNLL_IOB2.str.replace('I--ORG', 'I-ORG')


    sentence_index = df.sentence_id.unique().tolist()
    train, test_val = train_test_split(sentence_index, test_size=0.3, random_state=seed)
    test, val = train_test_split(test_val, test_size=0.5, random_state=seed)

    df_train = df.loc[df.sentence_id.isin(train), :]
    df_test = df.loc[df.sentence_id.isin(test), :]
    df_val = df.loc[df.sentence_id.isin(val), :]

    df_train.reset_index(inplace=True, drop=True)
    df_test.reset_index(inplace=True, drop=True)
    df_val.reset_index(inplace=True, drop=True)

    train_destination = p / 'aqmar_train.feather'
    test_destination = p / 'aqmar_test.feather'
    validation_destination = p / 'aqmar_validation.feather'

    df_train.to_feather(train_destination, compression='uncompressed')
    df_test.to_feather(test_destination, compression='uncompressed')
    df_val.to_feather(validation_destination, compression='uncompressed')
    print(f"processed ancora and saved to", train_destination, test_destination, validation_destination)

    train_details = {'corpus': 'aqmar', 
                    'subset': 'aqmar', 
                    'path': train_destination, 
                    'split': "train",
                    'language': 'ar', 
                    'tokens': len(df_train), 
                    'sentences': len(df_train.sentence_id.unique())}

    add_corpus(train_details)

    test_details = {'corpus': 'aqmar', 
                    'subset': 'aqmar', 
                    'path': test_destination, 
                    'split': "test",
                    'language': 'ar', 
                    'tokens': len(df_test), 
                    'sentences': len(df_test.sentence_id.unique())}

    add_corpus(test_details)

    validation_details = {'corpus': 'aqmar', 
                        'subset': 'aqmar', 
                        'path': validation_destination, 
                        'split': "validation",
                        'language': 'ar', 
                        'tokens': len(df_val), 
                        'sentences': len(df_val.sentence_id.unique())}

    add_corpus(validation_details)

    print('Done!')