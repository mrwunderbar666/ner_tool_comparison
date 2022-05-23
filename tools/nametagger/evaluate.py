import pandas as pd
from pathlib import Path
from datasets import load_metric

metric = load_metric("seqeval")

p = Path.cwd() / 'tools' / 'nametagger'
results_path = Path.cwd() / 'results' / 'nametagger.csv'
outputs = list(p.glob("tmp/*.feather"))

evaluations = []

for o in outputs:
    print('Evaluating', o)
    df = pd.read_feather(o)
    predictions = df.groupby('sentence_id')['nametagger'].agg(list).to_list()
    references = df.groupby('sentence_id')['references'].agg(list).to_list()
    assert all([len(p) == len(r) for p,r in zip(predictions, references)])
    results = metric.compute(predictions=predictions, references=references)
    r = [{'task': key, **val} for key, val in results.items() if type(val) == dict]
    overall = {k.replace('overall_', ''): v for k, v in results.items() if type(v) != dict}
    overall['task'] = 'overall'
    r.append(overall)
    r = pd.DataFrame(r)
    r['language'] = o.name.split('_')[-1].replace('.feather', '')
    r['corpus'] = o.name.split('_')[0]
    evaluations.append(r)


results_df = pd.concat(evaluations)

results_df.to_csv(results_path, index=False)
