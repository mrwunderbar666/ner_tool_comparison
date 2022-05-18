import sys
import zipfile
from pathlib import Path

sys.path.append(str(Path.cwd()))
from utils.downloader import downloader

p = Path.cwd() / 'tools' / 'corenlp'

tmp = p / 'tmp'

if not tmp.exists():
    tmp.mkdir()

main_package = "https://nlp.stanford.edu/software/stanford-corenlp-latest.zip"

languages = {
    "arabic": "https://search.maven.org/remotecontent?filepath=edu/stanford/nlp/stanford-corenlp/4.4.0/stanford-corenlp-4.4.0-models-arabic.jar",
    "chinese": "https://search.maven.org/remotecontent?filepath=edu/stanford/nlp/stanford-corenlp/4.4.0/stanford-corenlp-4.4.0-models-chinese.jar",
    "french": "https://search.maven.org/remotecontent?filepath=edu/stanford/nlp/stanford-corenlp/4.4.0/stanford-corenlp-4.4.0-models-french.jar",
    "german": "https://search.maven.org/remotecontent?filepath=edu/stanford/nlp/stanford-corenlp/4.4.0/stanford-corenlp-4.4.0-models-german.jar",
    "hungarian": "https://search.maven.org/remotecontent?filepath=edu/stanford/nlp/stanford-corenlp/4.4.0/stanford-corenlp-4.4.0-models-hungarian.jar",
    "italian": "https://search.maven.org/remotecontent?filepath=edu/stanford/nlp/stanford-corenlp/4.4.0/stanford-corenlp-4.4.0-models-italian.jar",
    "spanish": "https://search.maven.org/remotecontent?filepath=edu/stanford/nlp/stanford-corenlp/4.4.0/stanford-corenlp-4.4.0-models-spanish.jar"
}

downloader(main_package, tmp / 'stanford-corenlp-latest.zip')

z = zipfile.ZipFile(tmp / 'stanford-corenlp-latest.zip', mode='r')
z.extractall(path=p)

corenlp_folder = list(p.glob('stanford-corenlp-*'))[0]

for lang, url in languages.items():
    print(f'Downloading {lang}...')
    dest = url.split('/')[-1]
    downloader(url, corenlp_folder / dest)