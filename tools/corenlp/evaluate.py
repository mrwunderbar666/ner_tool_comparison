import sys
from pathlib import Path
import pandas as pd
import requests
from timeit import default_timer as timer
from datetime import timedelta
from time import sleep
from argparse import ArgumentParser

sys.path.insert(0, str(Path.cwd()))
from tools.corenlp.utils import launch_server, annotate
from utils.registry import load_registry
from utils.mappings import corenlp2conll
from utils.metrics import compute_metrics


argparser = ArgumentParser(prog='Run CoreNLP Evaluation')
argparser.add_argument('--debug', action='store_true', help='Debug flag (only test a random sample)')
args = argparser.parse_args()

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
registry = load_registry()
evaluations = []
results_path = Path.cwd() / 'results' / f'corenlp.csv'

for lang, language in languages.items():
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

    corpora = registry.loc[(registry.language == lang) & (registry.split == 'validation')]

    for _, row in corpora.iterrows():

        corpus_path = Path(row['path'])

        if not corpus_path.exists():
            print('could not find corpus:', corpus_path)
            continue

        print('Loading corpus:', corpus_path)

        df = pd.read_feather(corpus_path)
        df = df.loc[~df.token.isna(), :]

        if args.debug:
            import random
            sample_size = min(len(df.sentence_id.unique().tolist()), 100)
            sentende_ids = random.sample(df.sentence_id.unique().tolist(), sample_size)
            df = df.loc[df.sentence_id.isin(sentende_ids), :]
            df = df.reset_index(drop=True)

        print('Annotating...', corpus_path)
        start_validation = timer()

        annotate(df)
        end_validation = timer()
        validation_time = timedelta(seconds=end_validation-start_validation)

        df['corenlp_iob'] = df.corenlp_ner.replace(corenlp2conll)

        predictions = df.groupby('sentence_id')['corenlp_iob'].agg(list).to_list()
        references = df.groupby('sentence_id')['CoNLL_IOB2'].agg(list).to_list()
        
        results = compute_metrics(predictions, references)

        r = [{'task': key, **val} for key, val in results.items() if type(val) == dict]

        overall = {k.replace('overall_', ''): v for k, v in results.items() if type(v) != dict}
        overall['task'] = 'overall'

        r.append(overall)

        r = pd.DataFrame(r)
        r['language'] = lang
        r['corpus'] = row['corpus']
        r['subset'] = row['subset']
        r['validation_duration'] = validation_time.total_seconds()
        r['tokens'] = row['tokens']
        r['sentences'] = row['sentences']

        evaluations.append(r)

    corenlp_server.terminate()

results_df = pd.concat(evaluations)
results_df.to_csv(results_path, index=False)

print('Done!')