import sys
import json
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, str(Path.cwd()))
from utils.registry import add_corpus
from utils.mappings import ontonotes2conll

p = Path.cwd() / 'corpora' / 'ontonotes'

print('Loading json file...')
with open(p / 'ontonotes5_parsed.json', 'r') as f:
    j = json.load(f)


tokenized = []

available_splits = list(j.keys())

print('Flattening json to data frame...')

for key in available_splits:
    with tqdm(total=len(j[key]), unit='document') as pbar:
        for i, doc in enumerate(j[key]):
            iob = doc['tokens_iob']
            # assign document id to each list of tokens    
            iob = list(
                map(
                    lambda token: dict(token, 
                                       sentence_id=str(i).zfill(7), 
                                       language=doc['language'], 
                                       corpus='ontonotes', 
                                       subset=key), 
                    iob)
                )
            tokenized += iob
            pbar.update(1)

# free up some space
del j

df = pd.DataFrame(tokenized)
df.language = df.language.replace({'arabic': 'ar', 'chinese': 'zh', 'english': 'en'})

# Helper function: wraps .replace() method for multiprocessing
def replace_(series, d=ontonotes2conll):
    return series.replace(d, regex=True)

# Helper function to speed up the processing
# Runs a function in parallel
def parallel_process(series, func, num_partitions=16, num_cores=8):
   series_split = np.array_split(series, num_partitions)
   pool = Pool(num_cores)
   series = pd.concat(pool.map(func, series_split))
   pool.close()
   pool.join()
   return series

print('Converting annotations to CoNLL IOB2 format ...')
df['CoNLL_IOB2'] = parallel_process(df.IOB2, replace_)

for split in available_splits:
    print('saving corpus', split)
    tmp = df.loc[df.subset == split, :]

    corpus_destination = p / f'{split}.feather'

    corpus_details = {'corpus': 'ontonotes', 
                      'subset': split, 
                      'path': corpus_destination, 
                      'split': split.lower().replace('ing', ''),
                      'language': 'multi', 
                      'tokens': len(tmp), 
                      'sentences': len(tmp.sentence_id.unique())}

    add_corpus(corpus_details)
    tmp.reset_index(drop=True).to_feather(corpus_destination, compression='uncompressed')

    for lang in df.language.unique():
        corpus_destination = p / f'{lang}_{split}.feather'
        tmp_lang = tmp.loc[tmp.language == lang, :].reset_index(drop=True)
        corpus_details = {'corpus': 'ontonotes', 
                      'subset': f"{split}_{lang}", 
                      'path': corpus_destination, 
                      'split': split.lower().replace('ing', ''),
                      'language': lang, 
                      'tokens': len(tmp_lang), 
                      'sentences': len(tmp_lang.sentence_id.unique())}
        add_corpus(corpus_details)
        tmp_lang.to_feather(corpus_destination, compression='uncompressed')
    
print('Done!')
