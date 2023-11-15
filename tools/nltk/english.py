import sys
from pathlib import Path
import pandas as pd
from timeit import default_timer as timer
from datetime import timedelta
from tqdm import tqdm
from argparse import ArgumentParser

import nltk

sys.path.insert(0, str(Path.cwd()))
from utils.mappings import nltk2conll
from utils.registry import load_registry, male_female_registry
from utils.metrics import compute_metrics


argparser = ArgumentParser(prog='Run NLTK English Evaluation')
argparser.add_argument('--debug', action='store_true', help='Debug flag (only test a random sample)')
argparser.add_argument('--gender', action='store_true', help="If enabled, only test the gender specific subsets")
args = argparser.parse_args()

language = 'en'
p = Path.cwd() / 'tools' / 'nltk'
results_dir = Path.cwd() / 'results'

if args.gender:
    results_dir = Path.cwd() / 'results' / 'gender'

if not results_dir.exists():
    results_dir.mkdir(parents=True)

results_path = results_dir / f'nltk_{language}.csv'

registry = load_registry()

corpora = registry.loc[(registry.language == language) & (registry.split == 'validation')]

if args.gender:
    corpora = male_female_registry(corpora)

evaluations = []

print('Evaluating:', language)

for _, row in corpora.iterrows():

    path_corpus = Path(row['path'])

    if not path_corpus.exists():
        print('could not find corpus:', path_corpus)
        continue

    print('Loading corpus:', path_corpus)

    df = pd.read_feather(path_corpus)
    df = df.loc[~df.token.isna(), :]

    if args.debug:
        import random
        sample_size = min([len(df.sentence_id.unique().tolist()), 100])
        sentence_ids = random.sample(df.sentence_id.unique().tolist(), sample_size)
        df = df.loc[df.sentence_id.isin(sentence_ids), :]
        df = df.reset_index(drop=True)

    # ensure consistent order of sentences
    df.sentence_id = df.sentence_id.astype(str).str.zfill(6)

    start_validation = timer()
    print('Annotating...', path_corpus)
        
    sentences = df.groupby('sentence_id')['token'].agg(list)
    assert all(sentences.explode().index == df.sentence_id), 'IDs of sentences and dataframe do not align!'

    classified = []

    with tqdm(total=len(sentences), unit='sentence') as pbar:
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
            pbar.update(1)

    # rejoin with df

    df['nltk_ner'] = pd.Series(classified).explode().reset_index(drop=True)
    end_validation = timer()
    validation_time = timedelta(seconds=end_validation-start_validation)

    df['nltk_iob'] = df.nltk_ner.replace(nltk2conll)

    predictions = df.groupby('sentence_id')['nltk_iob'].agg(list).to_list()
    references = df.groupby('sentence_id')['CoNLL_IOB2'].agg(list).to_list() 

    results = compute_metrics(predictions=predictions, references=references)

    r = [{'task': key, **val} for key, val in results.items() if type(val) == dict]

    overall = {k.replace('overall_', ''): v for k, v in results.items() if type(v) != dict}
    overall['task'] = 'overall'

    r.append(overall)

    r = pd.DataFrame(r)
    r['language'] = language
    r['corpus'] = row['corpus']
    r['subset'] = row['subset']
    r['split'] = row['split']
    r['validation_duration'] = validation_time.total_seconds()
    r['tokens'] = row['tokens']
    r['sentences'] = row['sentences']

    evaluations.append(r)

results_df = pd.concat(evaluations)
if args.gender:
    results_df = results_df[results_df.task == 'PER'].reset_index(drop=True)

results_df.to_csv(results_path, index=False)

print('Done!')