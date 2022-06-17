import pandas as pd
from tqdm import tqdm

def parse_conll(f_path, columns=['token', 'lemma', 'pos', 'chunk', 'CoNLL_IOB2'], encoding='utf-8'):

    with open(f_path, 'r', encoding=encoding) as f:
        lines = f.readlines()
    
    doc_id = 0
    sentence_id = 0
    token_id = 1
    tokens = []
    
    with tqdm(total=len(lines), unit='line') as pbar:
        for l in lines:
            pbar.update()
            if l.startswith('-DOCSTART-'):
                doc_id += 1
                sentence_id = 0
                token_id = 1
                continue
            elif l.strip() == '':
                sentence_id += 1
                token_id = 1
                continue
            else:
                token = {k: v for k, v in zip(columns, l.split())}
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