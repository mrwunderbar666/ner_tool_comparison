import json
from pathlib import Path
import pandas as pd

p = Path.cwd()

infos = []

for j in p.glob('tools/xlmroberta/models_varying_b/*/model_infos.json'):
    with open(j) as f:
        infos.append(json.load(f))


df = pd.json_normalize(infos)

# main reason to use feather here is because then the file won't be recognized by glob *.csv commands
df.to_feather(p / 'results' / 'roberta_training_varying_size.feather', compression="uncompressed")
