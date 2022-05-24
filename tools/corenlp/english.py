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

language = 'english'
p = Path.cwd() / 'tools' / 'corenlp'
corenlp_folder = list(p.glob('stanford-corenlp-*'))[0]
results_path = Path.cwd() / 'results' / f'corenlp_{language}.csv'

corenlp_server = launch_server(corenlp_folder)

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

j = r.json()

corpora = {'conll': Path.cwd() / 'corpora' / 'conll' / 'conll2003_en_validation_iob.feather',
           'emerging': Path.cwd() / 'corpora' / 'emerging' / 'emerging.test.annotated.feather',
           'ontonotes': Path.cwd() / 'corpora' / 'ontonotes' / 'english_VALIDATION.feather',
           'wikiann': Path.cwd() / 'corpora' / 'wikiann' / 'wikiann-en_validation.feather'
           }

metric = load_metric("seqeval")
evaluations = []

print('Evaluating:', language)

for corpus, path_corpus in corpora.items():

    if not path_corpus.exists():
        print('could not find corpus:', corpus)
        continue

    print('Loading corpus:', corpus)

    df = pd.read_feather(path_corpus)
    df = df.loc[~df.token.isna(), ]

    start_validation = timer()
    print('Annotating...', corpus)

    annotate(df)
    end_validation = timer()
    validation_time = timedelta(seconds=end_validation-start_validation)

    df['corenlp_iob'] = df.corenlp_ner.replace(stanford2conll)

    predictions = df.groupby('sentence_id')['corenlp_iob'].agg(list).to_list()

    if 'CoNLL_IOB2' in df.columns:
        references = df.groupby('sentence_id')['CoNLL_IOB2'].agg(list).to_list()
    else:
        references = df.groupby('sentence_id')['IOB2'].agg(list).to_list()

    results = metric.compute(predictions=predictions, references=references)

    r = [{'task': key, **val} for key, val in results.items() if type(val) == dict]

    overall = {k.replace('overall_', ''): v for k, v in results.items() if type(v) != dict}
    overall['task'] = 'overall'

    r.append(overall)

    r = pd.DataFrame(r)
    r['language'] = language
    r['corpus'] = corpus
    r['validation_duration'] = validation_time.total_seconds()

    evaluations.append(r)

results_df = pd.concat(evaluations)

results_df.to_csv(results_path, index=False)
corenlp_server.terminate()