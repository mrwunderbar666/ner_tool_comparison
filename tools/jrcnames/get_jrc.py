import requests
from pathlib import Path
from tqdm import tqdm
import gzip
import pandas as pd
import sys
sys.path.insert(0, str(Path.cwd()))
from utils.downloader import downloader

print('Downloading JRC Names')

p = Path.cwd() / 'tools' / 'jrcnames'
tmp = p / 'tmp'

if not tmp.exists():
    tmp.mkdir()


url = "https://wt-public.emm4u.eu/data/entities.gzip"

downloader(url, tmp / 'entities.gzip')

print('Deflating data...')
with open(tmp / 'entities.gzip', 'rb') as f_in:
    decompressed = gzip.decompress(f_in.read())

with open(p / 'entities.tsv', 'wb') as f_out:
    f_out.write(decompressed)


df = pd.read_csv(p / 'entities.tsv', skiprows=1, sep='\t', header=None)
df.columns = ['id', 'type', 'lang', 'keyword']
df.to_feather(p / 'jrcnames.feather', compression="uncompressed")
