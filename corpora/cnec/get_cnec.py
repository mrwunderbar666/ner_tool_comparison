import requests
import zipfile
from pathlib import Path
from tqdm import tqdm
from bs4 import BeautifulSoup
import pandas as pd
import sys
sys.path.insert(0, str(Path.cwd()))
from utils.registry import add_corpus
from utils.downloader import downloader
from utils.mappings import cnec2conll

p = Path.cwd() / 'corpora' / 'cnec'
tmp = p / 'tmp'

if not tmp.exists():
    tmp.mkdir()

repo = "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11858/00-097C-0000-0023-1B22-8/Czech_Named_Entity_Corpus_2.0.zip"

print(f'Downloading Czech Named Entity Corpus 2.0 from: {repo}...')
downloader(repo, tmp / 'Czech_Named_Entity_Corpus_2.0.zip')
print('Success!')

z = zipfile.ZipFile(tmp / 'Czech_Named_Entity_Corpus_2.0.zip', mode='r')
z.extractall(path=tmp)


# helper function to unnest the token annotations
def recurse_children(node, tok_dict, level=0):
    for tag in node.find_all('children', recursive=False):
        a_ref = tag.find_all('a.rf', recursive=False)
        lm_top = tag.find_all('LM', recursive=False)
        if len(a_ref) >  0:
            for a in a_ref:
                if len(a.find_all('LM')) > 0:
                    for i, lm in enumerate(a.find_all('LM')):
                        tok_dict[lm.get_text()][f'NER_{level}'] = a.parent.ne_type.get_text()
                        tok_dict[lm.get_text()][f'position_NER_{level}'] = i
                else:
                    tok_dict[a.get_text()][f'NER_{level}'] = a.parent.ne_type.get_text()
                    tok_dict[a.get_text()][f'position_NER_{level}'] = 0
        elif len(lm_top) > 0:
            for lm in lm_top:
                if len(lm.find('a.rf').find_all('LM', recursive=False)) > 0:
                    for i, lm_sub in enumerate(lm.find('a.rf').find_all('LM', recursive=False)):
                        tok_dict[lm_sub.get_text()][f'NER_{level}'] = lm.ne_type.get_text()
                        tok_dict[lm_sub.get_text()][f'position_NER_{level}'] = i
                else:
                    tok_dict[lm.find('a.rf').get_text()][f'NER_{level}'] = lm.ne_type.get_text()
                    tok_dict[lm.find('a.rf').get_text()][f'position_NER_{level}'] = 0
                if len(lm.find_all('children', recursive=False)) > 0:
                    tok_dict = recurse_children(lm, tok_dict, level=level + 1)
        if len(tag.find_all('children', recursive=False)) > 0:
            tok_dict = recurse_children(tag, tok_dict, level=level + 1)
    return tok_dict 

print('Converting .treex to flat dataframes...')
# get all .treex files
for treex in tmp.glob('cnec2.0/data/treex/*.treex'):
    print('Converting:', treex, '...')

    with open(treex) as f:
        # use BeautifulSoup to parse XML files
        soup = BeautifulSoup(f, 'xml')

    # find all <LM> tags == Sentences
    sentences = soup.bundles.find_all("LM", recursive=False)
    print('Corpus has', len(sentences), 'sentences...')

    # initialize empty dict
    tokens = {}

    # iterate over every sentence and extract tokens
    print('Iterating over sentences...')
    with tqdm(total=len(sentences), unit='sentence') as pbar:
        for sentence in sentences:
            new_tokens = {lm['id']: 
                        {'sentence_id': sentence['id'],
                        'token_id': int(lm.ord.get_text()),
                        'token': lm.form.get_text(),
                        'cnec_token_id': lm['id'],
                        'position': lm.ord.get_text()} for lm in sentence.a_tree.find_all('LM', id=True)} 
            tokens.update(new_tokens)
            # recurse with helper function
            tokens = recurse_children(sentence.n_tree, tokens)
            pbar.update()

    # convert to pandas data frame
    print('Converting to data frame...')
    df = pd.DataFrame.from_dict(tokens, orient='index')
    df = df.reset_index(drop=True)
    

    # map to CoNLL IOB2 annotation style (see utils.mappings.py)
    # working from outer most to inner most tag
    # Challenge: e.g., Complex Address expression containing person names and locations
    # Ing. Tomáš BRYCHTA - SCANTRAVEL , Drtinova 17 , Praha 5
    # First we simply map from cnec to CoNLL-IOB2
    # P -> PER
    # p* -> PER
    # i* -> ORG
    # g* -> LOC
    # Then we replace all MISC with PER / LOC / ORG from the next lower level
    # we also get rid of numbers 

    for col in df.columns:
        if col.startswith('NER_'):
            df['IOB_' + col] = df[col].replace(cnec2conll, regex=True)
        

    for col in df.columns:
        if col.startswith('IOB_'):
            filt = (df['position_' + col.replace('IOB_', '')] == 0) & (df[col] != 'O')
            df.loc[filt, col] = 'B-' + df.loc[filt, col]
            filt = (df['position_' + col.replace('IOB_', '')] > 0) & (df[col] != 'O')
            df.loc[filt, col] = 'I-' + df.loc[filt, col]


    # filter: top level is MISC, Level below is NOT MISC and also NOT NA
    filt = (df.IOB_NER_0.str.contains('MISC', na=False)) & (df.IOB_NER_1.str.contains('MISC', na=False) == False) & (df.IOB_NER_1.notna())
    df.loc[filt, 'IOB_NER_0'] = df[filt].IOB_NER_1

    filt = (df.IOB_NER_0.str.contains('MISC', na=False)) & (df.IOB_NER_2.str.contains('MISC', na=False) == False) & (df.IOB_NER_2.notna())
    df.loc[filt, 'IOB_NER_0'] = df[filt].IOB_NER_2

    filt = (df.IOB_NER_0.str.contains('MISC', na=False)) & (df.IOB_NER_3.str.contains('MISC', na=False) == False) & (df.IOB_NER_3.notna())
    df.loc[filt, 'IOB_NER_0'] = df[filt].IOB_NER_3


    df['CoNLL_IOB2'] = df.IOB_NER_0
    df.loc[df.CoNLL_IOB2.isna(), 'CoNLL_IOB2'] = 'O'


    df = df.drop(df.filter(like='IOB_', axis=1).columns, axis=1)

    for col in df.columns:
        if col.startswith('NER_'):
            df[col.replace('NER_', 'CNEC_lvl_')] = df[col]
        if col.startswith('position_NER'):
            df[col.replace('NER', 'lvl')] = df[col]
        
    df['dataset'] = 'cnec2.0'
    df['subset'] = treex.name.replace('.treex', '')
    df['language'] = 'cs'

    cols = ["dataset", "language", "subset", "sentence_id", "token_id", "token", "CoNLL_IOB2", 'cnec_token_id', "position", "CNEC_lvl_0", "position_lvl_0", "CNEC_lvl_1",  
            "position_lvl_1", "CNEC_lvl_2", "position_lvl_2", "CNEC_lvl_3", "position_lvl_3"]

    df = df.loc[:, cols]
    df.sentence_id = df.sentence_id.str.replace('s', '').astype(int)
    df.sentence_id = df.sentence_id.astype(str).str.zfill(7)

    corpus_path =  p / treex.name.replace('.treex', '.feather')

    print('saving dataframe to ', corpus_path)
    df.to_feather(corpus_path, compression='uncompressed')

    split = ''
    if 'etest' in treex.name:
        split = 'validation'
    elif 'dtest' in treex.name:
        split = 'test'
    elif 'train' in treex.name:
        split = 'train'
    else:
        split = 'complete'

    corpus_details = {'corpus': 'cnec2.0',
                      'subset': treex.name.replace('.treex', ''), 
                      'path': corpus_path, 
                      'split': split, 
                      'language': 'cs', 
                      'tokens': len(df), 
                      'sentences': len(df.sentence_id.unique())}

    add_corpus(corpus_details)

print('Done!')