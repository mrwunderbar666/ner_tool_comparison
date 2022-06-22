import pandas as pd
from pathlib import Path
from datasets import load_metric
import sys
sys.path.insert(0, str(Path.cwd()))
from utils.mappings import nametagger2conll

metric = load_metric("seqeval")

p = Path.cwd() / 'tools' / 'nametagger'
results_path = Path.cwd() / 'results' / 'nametagger.csv'
outputs = list(p.glob("tmp/*.feather"))

evaluations = []

for o in outputs:
    print('Evaluating', o)
    df = pd.read_feather(o)
    if str(o).endswith('_cs.feather'):
        df.nametagger = df.nametagger.replace(nametagger2conll, regex = True)
        df.references = df.references.str.replace('I-', 'B-')
    predictions = df.groupby('sentence_id')['nametagger'].agg(list).to_list()
    references = df.groupby('sentence_id')['references'].agg(list).to_list()
    assert all([len(p) == len(r) for p,r in zip(predictions, references)])
    results = metric.compute(predictions=predictions, references=references)
    r = [{'task': key, **val} for key, val in results.items() if type(val) == dict]
    overall = {k.replace('overall_', ''): v for k, v in results.items() if type(v) != dict}
    overall['task'] = 'overall'
    r.append(overall)
    r = pd.DataFrame(r)

    meta_info = pd.read_csv(str(o).replace('.feather', '.csv'))
    r['corpus'] = meta_info.loc[0, 'corpus']
    r['subset'] = meta_info.loc[0, 'subset']
    r['language'] = meta_info.loc[0, 'language']
    r['sentences'] = meta_info.loc[0, 'sentences']
    r['tokens'] = meta_info.loc[0, 'tokens']
    r['evaluation_time'] = meta_info.loc[0, 'evaluation_time']
    evaluations.append(r)

results_df = pd.concat(evaluations)

results_df.to_csv(results_path, index=False)
