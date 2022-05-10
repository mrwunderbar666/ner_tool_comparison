import sys
import requests
from pathlib import Path
import pandas as pd

sys.path.append(str(Path.cwd()))
from utils.downloader import downloader

p = Path.cwd() / 'corpora' / 'germeval2014'
tmp = p / 'tmp'

if not tmp.exists():
    tmp.mkdir()

dev = "1ZfRcQThdtAR5PPRjIDtrVP7BtXSCUBbm"
test = "1u9mb7kNJHWQCWyweMDRMuTFoOHOfeBTH"
train = "1Jjhbal535VVz2ap4v4r_rN1UEHTdLK5P"

api = "https://drive.google.com/uc"

downloader(api, tmp / 'NER-de-dev.tsv', params={'id': dev})

downloader(api, tmp / 'NER-de-test.tsv', params={'id': test})

downloader(api, tmp / 'NER-de-train.tsv', params={'id': train})


print('parsing raw tsv files into dataframes ...')
for tsv in tmp.glob('*.tsv'):
    print('parsing:', tsv)
    corp = []
    corp_id = tsv.name.replace('.tsv', '')
    with open(tsv, 'r') as f_in:
        lines = f_in.readlines()
        sentence_id = 0
        sentence_source = ''
        sentence_date = ''
        for line in lines:
            line = line.split('\t')
            if line[0].startswith('#'):
                sentence_id +=1
                sentence_source = line[1].strip()
                sentence_date = line[2].strip().replace('[', '').replace(']', '')
                continue
            if line[0].startswith('\n'):
                continue
            corp.append({'dataset': 'germeval2014', 'language': 'de', 
                            'corpus': corp_id, 
                            'sentence_source': sentence_source,
                            'sentence_date': sentence_date,
                            'sentence': sentence_id, 
                            'token_id': line[0].strip(), 
                            'token': line[1].strip(), 
                            'BIO': line[2].strip(),
                            'BIO_nested': line[3].strip()})
        df = pd.DataFrame(corp)

        # add CoNLL IOB2 Format
        df['CoNLL_IOB2'] = df.BIO.str.extract(r'([BI]-[A-Z]{3})')
        df['CoNLL_IOB2'] = df.CoNLL_IOB2.str.replace('OTH', 'MISC')
        df.loc[df.CoNLL_IOB2.isna(), 'CoNLL_IOB2'] = 'O'

        print('successfully parsed!')
        df.to_feather(p / (corp_id + '.feather'), compression='uncompressed')
        print(f"saved to: {p / (corp_id + '.feather')}")
            
        
print('Done!')