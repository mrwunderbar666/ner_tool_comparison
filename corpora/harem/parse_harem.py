import sys
import typing

from pathlib import Path
from bs4 import BeautifulSoup
from bs4.element import Tag, NavigableString
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import pandas as pd

from nltk.tokenize import word_tokenize

sys.path.insert(0, str(Path.cwd()))
from utils.registry import add_corpus
from utils.mappings import harem2conll

seed = 5618

p = Path.cwd() / 'corpora' / 'harem'

def new_token(token, paragraph_id=None, iob = 'O', category=None, tipo=None) -> dict:
    return {'token': token, 'paragraph_id': paragraph_id, 'iob': iob, 'category': category, 'tipo': tipo}

def parse_paragraph(p: Tag, paragraph_id=1) -> typing.List[dict]:
    for omitido in p.find_all('OMITIDO'):
        omitido.unwrap()
    tokens = []
    for child in p.children:
        if type(child) == NavigableString:
            toks = word_tokenize(child.strip(), language="portuguese")
            tokens += [new_token(t, paragraph_id=paragraph_id) for t in toks]
        elif child.name == 'ALT':
            """
                The corpus offers alternative annotations for compund NE. 
                For example: "Martin Luther's 95 Theses" is annotated several times. 
                Once as a complete compound, and once as two separate entities (Martin Luther and the "95 Theses")

                The corpus format is not ergonomic, because the alternative annotations are just 
                separated with a plain '|' character
                `<EM ...>Martin Luther 95 Theses</EM> | <EM ...>Martin Luther</EM>'s <EM ...>95 Theses</EM></ALT>`
                
                This makes correct parsing rather convoluted. Therefore, the parser uses a simple implementation
                where always the first annotation (the complete compound) is retrieved. The alternatives are dropped.

            """
            toks = word_tokenize(child.EM.get_text(), language="portuguese")
            tmp_toks = [new_token(t, paragraph_id=paragraph_id, iob='I', category=child.EM.get('CATEG'), tipo=child.EM.get('TIPO')) for t in toks]
            tmp_toks[0]['iob'] = 'B'
            tokens += tmp_toks
        elif child.name == 'EM':
            toks = word_tokenize(child.get_text(), language="portuguese")
            tmp_toks = [new_token(t, paragraph_id=paragraph_id, iob='I', category=child.get('CATEG'), tipo=child.get('TIPO')) for t in toks]
            tmp_toks[0]['iob'] = 'B'
            tokens += tmp_toks
        else:
            raise ValueError
    return tokens

def parse_document(document: Tag) -> typing.List[dict]:
    doc_id = document['DOCID']
    tokens = []
    for i, p in enumerate(document.find_all('P'), start=1):
        paragraph = parse_paragraph(p, paragraph_id=i)
        for t in paragraph:
            t['document_id'] = doc_id
        tokens += paragraph
    return tokens
        



if __name__ == '__main__':

    print('Reading Harem XML file ...')
    with open(p / "CDSegundoHAREMReRelEM.xml", "rb") as f:
        soup = BeautifulSoup(f.read(), 'xml', from_encoding="ISO-8859-1")

    print('Parsing XML to tokens ...')
    tokens = []
    for doc in tqdm(soup.find_all('DOC')):
        tokens += parse_document(doc)

    print('Transform to dataframe and apply formatting ...')
    df = pd.DataFrame(tokens)

    df['ne_tag'] = df.category.fillna('').apply(lambda x: x.split('|')[0]).map(harem2conll).fillna('O')

    df['CoNLL_IOB2'] = df.ne_tag

    filt = df.ne_tag != 'O'
    df.loc[filt, 'CoNLL_IOB2'] = df[filt].iob + '-' + df[filt].CoNLL_IOB2

    df['sentence_id'] = df.document_id + '_' + df.paragraph_id.astype(str).str.zfill(3)
    df['corpus'] = 'HAREM'
    df['language'] = 'pt'

    df['token_id'] = df.groupby('sentence_id').cumcount() + 1

    cols = ['corpus', 'language', 'sentence_id', 'token_id', 'token', 'CoNLL_IOB2', 'category', 'tipo']

    df = df.loc[:,cols]

    sentence_index = df.sentence_id.unique().tolist()
    train, test_val = train_test_split(sentence_index, test_size=0.3, random_state=seed)
    test, val = train_test_split(test_val, test_size=0.5, random_state=seed)

    df_train = df.loc[df.sentence_id.isin(train), :]
    df_test = df.loc[df.sentence_id.isin(test), :]
    df_val = df.loc[df.sentence_id.isin(val), :]

    df_train.reset_index(inplace=True, drop=True)
    df_test.reset_index(inplace=True, drop=True)
    df_val.reset_index(inplace=True, drop=True)

    train_destination = p / f'harem_train.feather'
    test_destination = p / f'harem_test.feather'
    validation_destination = p / f'harem_validation.feather'

    df_train.to_feather(train_destination, compression='uncompressed')
    df_test.to_feather(test_destination, compression='uncompressed')
    df_val.to_feather(validation_destination, compression='uncompressed')
    print(f"processed HAREM and saved to", train_destination, test_destination, validation_destination)

    train_details = {'corpus': 'harem', 
                    'subset': f'HAREM', 
                    'path': train_destination, 
                    'split': "train",
                    'language': "pt", 
                    'tokens': len(df_train), 
                    'sentences': len(df_train.sentence_id.unique())}

    add_corpus(train_details)

    test_details = {'corpus': 'harem', 
                    'subset': f'HAREM', 
                    'path': test_destination, 
                    'split': "test",
                    'language': "pt", 
                    'tokens': len(df_test), 
                    'sentences': len(df_test.sentence_id.unique())}

    add_corpus(test_details)

    validation_details = {'corpus': 'harem', 
                        'subset': f'HAREM', 
                        'path': validation_destination, 
                        'split': "validation",
                        'language': "pt", 
                        'tokens': len(df_val), 
                        'sentences': len(df_val.sentence_id.unique())}

    add_corpus(validation_details)

    print('Done!')