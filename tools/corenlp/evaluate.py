import sys
from pathlib import Path
import pandas as pd
import requests
from datasets import load_metric
from timeit import default_timer as timer
from datetime import timedelta
from time import sleep

sys.path.insert(0, str(Path.cwd()))
from tools.corenlp.utils import launch_server, stanford2conll, annotate
from utils.registry import load_registry

import random

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
metric = load_metric("seqeval")
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

    # Send a test sentence first to provoke CoreNLP to load all files
    params = {'properties': '{"annotators":"ner","outputFormat":"json","tokenize.language": "Whitespace"}'}
    sentence = 'This is a testing sentence.'
    r = requests.post(server_address, params=params, data=sentence)

    assert r.status_code == 200, 'CoreNLP Server not responding!'

    corpora = registry.loc[(registry.language == lang) & (registry.split == 'validation')]
    # corpora = registry.loc[(registry.language == lang) & (registry.split == 'validation') & (registry.corpus == 'ontonotes')]

    for _, row in corpora.iterrows():

        corpus_path = Path(row['path'])

        if not corpus_path.exists():
            print('could not find corpus:', corpus_path)
            continue

        print('Loading corpus:', corpus_path)

        df = pd.read_feather(corpus_path)
        df = df.loc[~df.token.isna(), ]

        # for debugging, make the sample smaller
        # sentence_index = df.sentence_id.unique().tolist()
        # if len(sentence_index) > 20000:
        #     sentence_index = random.sample(sentence_index, 20000)
        #     df = df.loc[df.sentence_id.isin(sentence_index)]

        print('Annotating...', corpus_path)
        start_validation = timer()

        annotate(df)
        end_validation = timer()
        validation_time = timedelta(seconds=end_validation-start_validation)

        df['corenlp_iob'] = df.corenlp_ner.replace(stanford2conll)

        predictions = df.groupby('sentence_id')['corenlp_iob'].agg(list).to_list()
        references = df.groupby('sentence_id')['CoNLL_IOB2'].agg(list).to_list()
        
        results = metric.compute(predictions=predictions, references=references)

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