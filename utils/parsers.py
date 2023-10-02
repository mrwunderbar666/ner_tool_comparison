import pandas as pd
from tqdm import tqdm
from pathlib import Path
from typing import Union

def parse_conll(f_path: Union[Path, str], 
                columns=['token', 'lemma', 'pos', 'chunk', 'CoNLL_IOB2'], 
                encoding='utf-8', 
                separator=' ') -> pd.DataFrame:
    """
        Generic CoNLL IOB format parser

        CoNLL format comes without column header (so need to pass in separately)
    """

    with open(f_path, 'r', encoding=encoding) as f:
        lines = f.readlines()
    
    doc_id = 0
    sentence_id = 0
    token_id = 1
    tokens = []
    
    with tqdm(total=len(lines), unit='line') as pbar:
        for l in lines:
            pbar.update()
            if l.startswith('-DOCSTART-') or l.startswith('<HEADLINE>'):
                doc_id += 1
                sentence_id = 0
                token_id = 1
                continue
            elif l.strip() in ['<BODY>', '<INGRESS>']:
                # FiNER uses these tags to divide documents
                continue
            elif l.strip() == '':
                sentence_id += 1
                token_id = 1
                continue
            else:
                token = {k.strip(): v.strip() for k, v in zip(columns, l.split(separator))}
                token['doc_id'] = doc_id
                token['sentence_id'] = sentence_id
                token['token_id'] = token_id
                tokens.append(token)
                token_id += 1
    
    df = pd.DataFrame(tokens)
    # sometimes there is not blank line after -DOCSTART-
    if min(df.sentence_id) == 0:
        df.sentence_id = df.sentence_id + 1
            
    return df


def parse_hipe(f_path: Path) -> pd.DataFrame:
    """
        HIPE Dialect of CoNLL IOB Format
    
        Contains richer metainformation that are provided as comments 
        (lines start with `#`)

    """
    with open(f_path, 'r') as f:
        lines = f.readlines()

    document_number = 1
    sentence_id = 1
    token_id = 1
    language = ""
    newspaper = ""
    date = ""
    document_id = ""
    tokens = []

    columns = lines.pop(0).split('\t')
    columns = [c.strip() for c in columns]

    for l in lines:
        if l.lower().startswith('# language'):
            language = l.split('=')[-1].strip()
            continue
        elif l.lower().startswith('# newspaper'):
            newspaper = l.split('=')[-1].strip()
            continue
        elif l.lower().startswith('# date'):
            date = l.split('=')[-1].strip()
            continue
        elif l.lower().startswith('# document_id'):
            document_id = l.split('=')[-1].strip()
            continue
        elif l.strip() == '':
            document_number += 1
            sentence_id += 1
            token_id = 1
            continue
        else:
            token = {k.strip(): v.strip() for k, v in zip(columns, l.split("\t"))}
            token['document_number'] = document_number
            token['sentence_id'] = sentence_id
            token['token_id'] = token_id
            token['language'] = language
            token['newspaper'] = newspaper
            token['date'] = date
            token['document_id'] = document_id
            tokens.append(token)
            token_id += 1

    return pd.DataFrame(tokens)



def parse_conllup(f_path: Path) -> pd.DataFrame:
    """
        CoNLL-U+ format

        has a column declaration at the header

    """
    with open(f_path, 'r') as f:
        lines = f.readlines()

    sentence_id = 1
    token_id = 1
    tokens = []

    columns = lines.pop(0).split('=')[-1].split()
    columns = [c.strip() for c in columns]

    for l in lines:
        if l.strip() == '':
            sentence_id += 1
            token_id = 1
            continue
        else:
            token = {k.strip(): v.strip() for k, v in zip(columns, l.split("\t"))}
            token['sentence_id'] = sentence_id
            token['token_id'] = token_id
            tokens.append(token)
            token_id += 1

    return pd.DataFrame(tokens)