import json
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import numpy as np
from multiprocessing import Pool


p = Path.cwd() / 'corpora' / 'ontonotes'

with open(p / 'ontonotes5_parsed.json', 'r') as f:
    j = json.load(f)


tokenized = []

available_splits = list(j.keys())

for key in available_splits:
    with tqdm(total=len(j[key]), unit='document') as pbar:
        for i, doc in enumerate(j[key]):
            iob = doc['tokens_iob']
            # assign document id to each list of tokens    
            iob = list(map(lambda token: dict(token, doc_id=i, language=doc['language'], split=key), iob))
            tokenized += iob
            pbar.update(1)

del j

df = pd.DataFrame(tokenized)

iob2conll = {'I-PERSON': 'I-PER', 'B-PERSON': 'B-PER',
             'I-GPE': 'I-LOC', 'B-GPE': 'B-LOC',
             'I-FAC': 'I-LOC', 'B-FAC': 'B-LOC',
             'I-EVENT': 'I-MISC', 'B-EVENT': 'B-MISC', 
             'I-WORK_OF_ART': 'I-MISC', 'B-WORK_OF_ART': 'B-MISC', 
             'I-PRODUCT': 'I-MISC', 'B-PRODUCT': 'B-MISC', 
             'I-LAW': 'I-MISC', 'B-LAW': 'B-MISC', 
             '[BI]-NORP': 'O', 
             '[BI]-LANGUAGE': 'O', 
             '[BI]-DATE': 'O', 
             '[BI]-TIME': 'O', 
             '[BI]-CARDINAL': 'O', 
             '[BI]-MONEY': 'O', 
             '[BI]-PERCENT': 'O', 
             '[BI]-ORDINAL': 'O', 
             '[BI]-QUANTITY': 'O'}

# Helper function: wraps .replace() method for multiprocessing
def replace_(series, d=iob2conll):
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

df['CoNLL_IOB2'] = parallel_process(df.IOB2, replace_)

for split in available_splits:
    tmp = df.loc[df.split == split, :]
    tmp.reset_index().to_feather(p / f'{split}.feather', compression='uncompressed')
    for lang in df.language.unique():
        tmp.loc[tmp.language == lang, :].reset_index().to_feather(p / f'{lang}_{split}.feather', compression='uncompressed')
    


