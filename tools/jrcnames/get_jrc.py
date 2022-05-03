import requests

import yaml

from pathlib import Path
from tqdm import tqdm
import gzip
import pandas as pd
import shutil

def downloader(response, destination):
    total_size = int(response.headers.get('content-length', 0))
    with open(destination, 'wb') as f:
        with tqdm(total=total_size) as pbar:
            for chunk in response.iter_content(chunk_size=1024):
                b = f.write(chunk)
                pbar.update(b)

p = Path.cwd() / 'tools' / 'jrcnames'
tmp = p / 'tmp'

if not tmp.exists():
    tmp.mkdir()


url = "https://wt-public.emm4u.eu/data/entities.gzip"

r = requests.get(url, stream=True)
downloader(r, tmp / 'entities.gzip')

print('Deflating data...')
with open(tmp / 'entities.gzip', 'rb') as f_in:
    decompressed = gzip.decompress(f_in.read())

with open(p / 'entities.tsv', 'wb') as f_out:
    f_out.write(decompressed)


df = pd.read_csv(p / 'entities.tsv', skiprows=1, sep='\t', header=None)
df.columns = ['id', 'type', 'lang', 'keyword']
df.to_feather(p / 'jrcnames.feather', compression="uncompressed")

shutil.rmtree(tmp)