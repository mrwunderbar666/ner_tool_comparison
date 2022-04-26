import requests
import zipfile
from pathlib import Path
import shutil
from tqdm import tqdm
from nltk.corpus.reader import XMLCorpusReader

p = Path.cwd() / 'datasets' / 'cnec'
tmp = p / 'tmp'

def downloader(response, destination):
    total_size = int(response.headers.get('content-length', 0))
    with open(destination, 'wb') as f:
        with tqdm(total=total_size) as pbar:
            for chunk in response.iter_content(chunk_size=1024):
                b = f.write(chunk)
                pbar.update(b)

if not tmp.exists():
    tmp.mkdir()

repo = "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11858/00-097C-0000-0023-1B22-8/Czech_Named_Entity_Corpus_2.0.zip"

print(f'Downloading Czech Named Entity Corpus 2.0 from: {repo}...')
r = requests.get(repo, stream=True)
downloader(r, tmp / 'Czech_Named_Entity_Corpus_2.0.zip')
print('Success!')

z = zipfile.ZipFile(tmp / 'Czech_Named_Entity_Corpus_2.0.zip', mode='r')
z.extractall(path=tmp)