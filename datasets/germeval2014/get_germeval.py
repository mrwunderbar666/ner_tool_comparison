import requests
from pathlib import Path
import pandas as pd
from nltk.corpus.reader import ConllChunkCorpusReader
import shutil
from tqdm import tqdm

def downloader(response, destination):
    total_size = int(response.headers.get('content-length', 0))
    with open(destination, 'wb') as f:
        with tqdm(total=total_size) as pbar:
            for chunk in r.iter_content(chunk_size=1024):
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