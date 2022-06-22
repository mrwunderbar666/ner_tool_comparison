import sys
from pathlib import Path
import pandas as pd
from datasets import load_metric
from timeit import default_timer as timer
from datetime import timedelta

sys.path.insert(0, str(Path.cwd()))
from tools.opennlp.opennlp import annotate
from utils.mappings import opennlp2conll
from utils.registry import load_registry

languages = ['en', 'nl', 'es']

p = Path.cwd() / 'tools' / 'opennlp'
opennlp_dir = list(p.glob('apache-opennlp-*'))[0]
opennlp_bin = opennlp_dir / 'bin' / 'opennlp'

results_path = Path.cwd() / 'results' / f'opennlp.csv'
registry = load_registry()


metric = load_metric("seqeval")
evaluations = []

for language in languages:

    print('Evaluating:', language)
    models = {'person': p / 'models' / f'{language}-ner-person.bin',
            'organization': p / 'models' / f'{language}-ner-organization.bin',
            'location': p / 'models' / f'{language}-ner-location.bin',
            'misc': p / 'models' / f'{language}-ner-misc.bin'}

    corpora = registry.loc[(registry.language == language) & (registry.split == 'validation')]

    for _, row in corpora.iterrows():

        path_corpus = Path(row['path'])

        if not path_corpus.exists():
            print('could not find corpus:', path_corpus)
            continue

        print('Loading corpus:', path_corpus)

        df = pd.read_feather(path_corpus)
        df = df.loc[~df.token.isna(), ]

        start_validation = timer()
        print('Annotating...', path_corpus)
            
        sentences = df.groupby('sentence_id')['token'].agg(list).tolist()
        sentences = "\n".join([" ".join(s) for s in sentences])

        for model, model_path in models.items():
            if not model_path.exists(): continue
            tagged = annotate(sentences, opennlp_bin, model=model_path)
            tagged = pd.Series(tagged).explode().reset_index(drop=True)
            df[model] = tagged
        end_validation = timer()
        validation_time = timedelta(seconds=end_validation-start_validation)

        df['opennlp_ner'] = 'O'

        for model in models.keys():
            if model not in df.columns: continue
            filt = df['opennlp_ner'] == 'O'
            df.loc[filt, 'opennlp_ner'] = df.loc[filt, model]

        df['opennlp_ner'] = df.opennlp_ner.replace(opennlp2conll)

        predictions = df.groupby('sentence_id')['opennlp_ner'].agg(list).to_list()
        references = df.groupby('sentence_id')['CoNLL_IOB2'].agg(list).to_list()
        
        results = metric.compute(predictions=predictions, references=references)

        r = [{'task': key, **val} for key, val in results.items() if type(val) == dict]

        overall = {k.replace('overall_', ''): v for k, v in results.items() if type(v) != dict}
        overall['task'] = 'overall'

        r.append(overall)

        r = pd.DataFrame(r)
        r['language'] = language
        r['corpus'] = row['corpus']
        r['subset'] = row['subset']
        r['validation_duration'] = validation_time.total_seconds()
        r['tokens'] = row['tokens']
        r['sentences'] = row['sentences']

        evaluations.append(r)

results_df = pd.concat(evaluations)
results_df.to_csv(results_path, index=False)

print('Done!')