import sys
from pathlib import Path
import pandas as pd
import requests
from time import sleep

sys.path.insert(0, str(Path.cwd()))
from tools.corenlp.utils import launch_server, annotate_sentence
from utils.challenges import load_challenges

languages = {'zh': 'chinese', 
             'en': 'english', 
             'fr': 'french', 
             'de': 'german', 
             'hu': 'hungarian',
             'it': 'italian',
             'es': 'spanish'
             }

p = Path.cwd() / 'tools' / 'corenlp'

corenlp_folder = list(p.glob('stanford-corenlp-*'))[0]
results_path = Path.cwd() / 'results' / 'corenlp_challenges.json'

challenges = load_challenges()

challenges['tool'] = 'corenlp'
challenges['tokens'] = ''
challenges['iob'] = ''


for lang, language in languages.items():
    if lang not in challenges.language.unique():
        continue
    print('Evaluating language:', language)
    
    corenlp_server = launch_server(corenlp_folder, language=language)
    corenlp_ready = False
    server_address = 'http://localhost:9000/'

    while not corenlp_ready:
        try:
            r = requests.get(server_address + 'ready')
            if r.status_code == 200:
                corenlp_ready = True
        except:
            print('waiting for server...')
        finally:
            sleep(0.5)

    # Send a test sentence to provoke CoreNLP to load all files
    params = {'properties': '{"annotators":"ner","outputFormat":"json","tokenize.language": "Whitespace"}'}
    sentence = 'This is a testing sentence.'
    r = requests.post(server_address, params=params, data=sentence)

    assert r.status_code == 200, 'CoreNLP Server not responding!'

    challenge_sentences = challenges.loc[(challenges.language == lang)]
    for index, row in challenge_sentences.iterrows():

        tokens, iob = annotate_sentence(row['text'])
        challenges.at[index, 'tokens'] = tokens
        challenges.at[index, 'iob'] = iob


    corenlp_server.terminate()


challenges.to_json(results_path, orient="records")

print('Done!')