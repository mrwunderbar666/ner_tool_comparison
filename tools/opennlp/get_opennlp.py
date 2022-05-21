import sys
from pathlib import Path
import tarfile

sys.path.append(str(Path.cwd()))
from utils.downloader import downloader


p = Path.cwd() / 'tools' / 'opennlp'

tmp = p / 'tmp'
models_dir = p / 'models'

if not tmp.exists():
    tmp.mkdir()

if not models_dir.exists():
    models_dir.mkdir()


opennlp = "https://dlcdn.apache.org/opennlp/opennlp-1.9.4/apache-opennlp-1.9.4-bin.tar.gz"

print('Downloading Apache Open NLP...')

downloader(opennlp, tmp / opennlp.split('/')[-1])

print('Extracting archive...')
t = tarfile.open(tmp / opennlp.split('/')[-1])
t.extractall(p)

print('Success!')

models_base_url = "http://opennlp.sourceforge.net/models-1.5/"

models = ["en-ner-date.bin", 
            "en-ner-location.bin", 
            "en-ner-money.bin", 
            "en-ner-organization.bin", 
            "en-ner-percentage.bin", 
            "en-ner-person.bin", 
            "en-ner-time.bin",
            "es-ner-person.bin",
            "es-ner-organization.bin",
            "es-ner-location.bin",
            "es-ner-misc.bin",
            "nl-ner-person.bin",
            "nl-ner-organization.bin",
            "nl-ner-location.bin",
            "nl-ner-misc.bin"]

for model in models:
    print('Downloading:', model)
    downloader(models_base_url + model, models_dir / model)

print('Done!')