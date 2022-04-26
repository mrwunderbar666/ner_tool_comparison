import requests
from pathlib import Path
import pandas as pd
import shutil
from tqdm import tqdm

def downloader(response, destination):
    total_size = int(response.headers.get('content-length', 0))
    with open(destination, 'wb') as f:
        with tqdm(total=total_size) as pbar:
            for chunk in response.iter_content(chunk_size=1024):
                b = f.write(chunk)
                pbar.update(b)


p = Path.cwd() / 'datasets' / 'germeval2014'
tmp = p / 'tmp'

if not tmp.exists():
    tmp.mkdir()

dev = "1ZfRcQThdtAR5PPRjIDtrVP7BtXSCUBbm"
test = "1u9mb7kNJHWQCWyweMDRMuTFoOHOfeBTH"
train = "1Jjhbal535VVz2ap4v4r_rN1UEHTdLK5P"

api = "https://drive.google.com/uc"

r = requests.get(api, params={'id': dev}, stream=True)
downloader(r, tmp / 'NER-de-dev.tsv')

r = requests.get(api, params={'id': test}, stream=True)
downloader(r, tmp / 'NER-de-test.tsv')

r = requests.get(api, params={'id': train}, stream=True)
downloader(r, tmp / 'NER-de-train.tsv')


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
        print('successfully parsed!')
        df.to_feather(p / (corp_id + '.feather'), compression='uncompressed')
        print(f"saved to: {p / (corp_id + '.feather')}")
            
        
shutil.rmtree(tmp)
print('Done!')