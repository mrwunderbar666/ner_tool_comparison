import sys
from pathlib import Path
import pandas as pd
from datasets import load_metric
from timeit import default_timer as timer
from datetime import timedelta
from time import sleep
from tqdm import tqdm


sys.path.insert(0, str(Path.cwd()))
from utils.registry import load_registry
from utils.mappings import sonar2conll

from frog import Frog, FrogOptions


languages = {'nl': 'dutch'}

p = Path.cwd() / 'tools' / 'frog'

registry = load_registry()
metric = load_metric("seqeval")
evaluations = []
results_path = Path.cwd() / 'results' / 'frog_challenges.json'


challenges = pd.read_json(Path.cwd() / 'challenges.json')

challenges['tool'] = 'frog'
challenges['tokens'] = ''
challenges['iob'] = ''


frog = Frog(FrogOptions(parser=False))

challenge_sentences = challenges.loc[(challenges.language == 'nl')]
for index, row in challenge_sentences.iterrows():
    result = frog.process(row['text'])
    iob = [token['ner'] for token in result]
    tokens = [token['text'] for token in result]
    challenges.at[index, 'tokens'] = tokens
    challenges.at[index, 'iob'] = iob


challenges.to_json(results_path, orient="records")

print('Done!')