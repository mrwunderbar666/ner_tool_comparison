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
from utils.mappings import ancora2conll

seed = 5618

def parse_xml(path: Path) -> t.List[dict]:
    with open(path) as f:
        soup = BeautifulSoup(f.read(), 'xml')

    document_id = path.parent.name + '_' + path.name.replace('.xml', '')

    article = []
    language = soup.article['lng']
    for i, sentence in enumerate(soup.find_all('sentence'), start=1):
        s = [token for token in sentence.find_all(wd=True)]
        for token in s:
            token['language'] = language
            token['sentence_number'] = i
            token['document_id'] = document_id
            bio = 'B'
            token['bio'] = bio
            if "_" in token['wd']:
                for subtoken in token['wd'].split('_'):
                    newtoken = copy.copy(token)
                    newtoken['wd'] = subtoken
                    newtoken['bio'] = bio
                    article.append(newtoken.attrs)
                    bio = 'I'
            else:
                article.append(token.attrs)
    
    return article

if __name__ == '__main__':

    p = Path.cwd() / 'corpora' / 'ancora'

    tmp = p / 'tmp'

    if not tmp.exists():
        tmp.mkdir(parents=True)

    archives = p.glob('*.zip')
    for archive in archives:

        print('Extracting', archive.name, '...')
        z = zipfile.ZipFile(archive, mode='r')
        z.extractall(path=tmp)

    xml_files = list(tmp.glob('*/*/*.xml'))

    corpus = []

    assert len(xml_files) > 0

    print('Parsing XML files ...')

    with tqdm(total=len(xml_files)) as pbar:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(parse_xml, xml): xml for xml in xml_files}
            for future in as_completed(futures):
                xml = futures[future]
                try:
                    corpus += future.result()
                    pbar.update(1)
                except Exception as e:
                    print(f'{xml.name} generated an exception:', e)


    df = pd.DataFrame(corpus)

    df['CoNLL_IOB2'] = df['bio'].astype(str) + '-' + df['ne'].replace(ancora2conll)
    df.loc[~df['ne'].isin(ancora2conll.keys()), 'CoNLL_IOB2'] = 'O'

    df['sentence_id'] = df.document_id + '_' + df.sentence_number.astype(str).str.zfill(4)
    df['token_id'] = df.groupby('sentence_id').cumcount() + 1

    df = df.rename(columns={'wd': 'token', 'lem': 'lemma'})
    df['corpus'] = 'ancora'

    cols = ['corpus', 'language', 'sentence_id', 'token_id', 'token', 'lemma', 'CoNLL_IOB2']

    df = df.loc[:, cols]

    for language in df['language'].unique().tolist():
        print('Exporting language:', language)
        subset = df.loc[df.language == language]

        sentence_index = subset.sentence_id.unique().tolist()
        train, test_val = train_test_split(sentence_index, test_size=0.3, random_state=seed)
        test, val = train_test_split(test_val, test_size=0.5, random_state=seed)

        df_train = subset.loc[subset.sentence_id.isin(train), :]
        df_test = subset.loc[subset.sentence_id.isin(test), :]
        df_val = subset.loc[subset.sentence_id.isin(val), :]

        df_train.reset_index(inplace=True, drop=True)
        df_test.reset_index(inplace=True, drop=True)
        df_val.reset_index(inplace=True, drop=True)

        train_destination = p / f'ancora_{language}_train.feather'
        test_destination = p / f'ancora_{language}_test.feather'
        validation_destination = p / f'ancora_{language}_validation.feather'

        df_train.to_feather(train_destination, compression='uncompressed')
        df_test.to_feather(test_destination, compression='uncompressed')
        df_val.to_feather(validation_destination, compression='uncompressed')
        print(f"processed ancora and saved to", train_destination, test_destination, validation_destination)

        train_details = {'corpus': 'ancora', 
                        'subset': f'ancora-{language}-2.0', 
                        'path': train_destination, 
                        'split': "train",
                        'language': language, 
                        'tokens': len(df_train), 
                        'sentences': len(df_train.sentence_id.unique())}

        add_corpus(train_details)

        test_details = {'corpus': 'ancora', 
                        'subset': f'ancora-{language}-2.0', 
                        'path': test_destination, 
                        'split': "test",
                        'language': language, 
                        'tokens': len(df_test), 
                        'sentences': len(df_test.sentence_id.unique())}

        add_corpus(test_details)

        validation_details = {'corpus': 'ancora', 
                            'subset': f'ancora-{language}-2.0', 
                            'path': validation_destination, 
                            'split': "validation",
                            'language': language, 
                            'tokens': len(df_val), 
                            'sentences': len(df_val.sentence_id.unique())}

        add_corpus(validation_details)

    print('Done!')