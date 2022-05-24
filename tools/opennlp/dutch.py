import sys
from pathlib import Path
import pandas as pd
from datasets import load_metric
from timeit import default_timer as timer
from datetime import timedelta

import nltk

sys.path.insert(0, str(Path.cwd()))
from tools.opennlp.opennlp import annotate, apache2conll

language = 'nl'
p = Path.cwd() / 'tools' / 'opennlp'
opennlp_dir = list(p.glob('apache-opennlp-*'))[0]
opennlp_bin = opennlp_dir / 'bin' / 'opennlp'
models = {'person': p / 'models' / f'{language}-ner-person.bin',
          'organization': p / 'models' / f'{language}-ner-organization.bin',
          'location': p / 'models' / f'{language}-ner-location.bin',
          'misc': p / 'models' / f'{language}-ner-misc.bin'}

results_path = Path.cwd() / 'results' / f'opennlp_{language}.csv'

corpora = {'ned.testb': Path.cwd() / 'corpora' / 'conll' / 'ned.testb.feather',
           'enp_NL.kb': Path.cwd() / 'corpora' / 'europeana' / 'enp_NL.kb_validation.feather',
           'wikiann': Path.cwd() / 'corpora' / 'wikiann' / 'wikiann-nl_validation.feather'
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


    # ensure consistent order of sentences
    df.sentence_id = df.sentence_id.astype(str).str.zfill(6)

    if corpus == 'ned.testb':
            df.sentence_id = df.doc + '_' + df.sentence_id

    start_validation = timer()
    print('Annotating...', corpus)
        
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


    df['opennlp_ner'] = df.opennlp_ner.replace(apache2conll)

    predictions = df.groupby('sentence_id')['opennlp_ner'].agg(list).to_list()

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
