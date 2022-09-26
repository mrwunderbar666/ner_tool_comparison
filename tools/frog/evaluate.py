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
results_path = Path.cwd() / 'results' / f'frog.csv'


frog = Frog(FrogOptions(parser=False, tok=False))

corpora = registry.loc[(registry.language == 'nl') & (registry.split == 'validation')]

for _, row in corpora.iterrows():
    corpus_path = row['path']

    df = pd.read_feather(corpus_path)
    df = df.loc[~df.token.isna(), ]

    df['frog_ner'] = 'O'
    print('Annotating...', corpus_path)
    start_validation = timer()
    with tqdm(df['sentence_id'].unique(), unit="sentence") as pbar:
        for sentence_id in df['sentence_id'].unique():
            filt_sentence = (df['sentence_id'] == sentence_id)
            sentence = " ".join(df.loc[filt_sentence, 'token'].tolist())
            result = frog.process(sentence)
            iob = [token['ner'] for token in result]
            assert len(iob) == len(df.loc[filt_sentence, 'frog_ner'])
            df.loc[filt_sentence, 'frog_ner'] = iob
            pbar.update(1)

    end_validation = timer()
    validation_time = timedelta(seconds=end_validation-start_validation)

    df['frog_ner'] = df.frog_ner.str.upper()
    df['frog_ner'] = df.frog_ner.replace(sonar2conll)

    predictions = df.groupby('sentence_id')['frog_ner'].agg(list).to_list()
    references = df.groupby('sentence_id')['CoNLL_IOB2'].agg(list).to_list()

    results = metric.compute(predictions=predictions, references=references)

    r = [{'task': key, **val} for key, val in results.items() if type(val) == dict]

    overall = {k.replace('overall_', ''): v for k, v in results.items() if type(v) != dict}
    overall['task'] = 'overall'

    r.append(overall)

    r = pd.DataFrame(r)
    r['language'] = 'nl'
    r['corpus'] = row['corpus']
    r['subset'] = row['subset']
    r['validation_duration'] = validation_time.total_seconds()
    r['tokens'] = row['tokens']
    r['sentences'] = row['sentences']

    evaluations.append(r)


results_df = pd.concat(evaluations)
results_df.to_csv(results_path, index=False)

print('Done!')