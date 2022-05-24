import sys
from pathlib import Path
import pandas as pd
from datasets import load_metric
from timeit import default_timer as timer
from datetime import timedelta

import nltk

sys.path.insert(0, str(Path.cwd()))
from tools.nltk.utils import nltk2conll

language = 'english'
p = Path.cwd() / 'tools' / 'nltk'
results_path = Path.cwd() / 'results' / f'nltk_{language}.csv'

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


    # ensure consistent order of sentences
    df.sentence_id = df.sentence_id.astype(str).str.zfill(6)

    start_validation = timer()
    print('Annotating...', corpus)

        
    sentences = df.groupby('sentence_id')['token'].agg(list).tolist()

    classified = []

    for sentence in sentences:
        pos = nltk.pos_tag(sentence)
        ner = nltk.ne_chunk(pos)
        tagged = []
        for token in ner:
            if isinstance(token, tuple):
                tagged.append('O')
            else:
                first = True
                for t in token:
                    tagged.append('B-' + token.label()) if first else tagged.append('I-' + token.label())
                    first = False
        classified.append(tagged)

    # rejoin with df
    df['nltk_ner'] = pd.Series(classified).explode().reset_index(drop=True)
    end_validation = timer()
    validation_time = timedelta(seconds=end_validation-start_validation)

    df['nltk_iob'] = df.nltk_ner.replace(nltk2conll)

    predictions = df.groupby('sentence_id')['nltk_iob'].agg(list).to_list()

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
