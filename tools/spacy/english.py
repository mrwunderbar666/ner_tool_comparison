import sys
import re
import spacy
from spacy.tokenizer import Tokenizer
from pathlib import Path
import pandas as pd
from timeit import default_timer as timer
from datetime import timedelta


from datasets import load_metric

from tqdm import tqdm
sys.path.append(str(Path.cwd()))
from tools.spacy.helpers import spacy2conll

language = 'en'
p = Path.cwd() / 'tools' / 'spacy'



results_path = Path.cwd() / 'results' / f'spacy_{language}.csv'

corpora = {'conll': Path.cwd() / 'corpora' / 'conll' / 'conll2003_en_validation_iob.feather',
           #'emerging': Path.cwd() / 'corpora' / 'emerging' / 'emerging.test.annotated.feather',
           #'ontonotes': Path.cwd() / 'corpora' / 'ontonotes' / 'english_VALIDATION.feather',
           #'wikiann': Path.cwd() / 'corpora' / 'wikiann' / 'wikiann-en_validation.feather'
           }

metric = load_metric("seqeval")
evaluations = []


nlp = spacy.load("en_core_web_sm", disable=["parser"])
# custom tokenizer
# conll datasets do not split hyphenated words
nlp.tokenizer = Tokenizer(nlp.vocab, token_match=re.compile(r'\S+').match)


for corpus, path_corpus in corpora.items():

    if not path_corpus.exists():
        print('could not find corpus:', corpus)
        continue

    print('Loading corpus:', corpus)

    df = pd.read_feather(path_corpus)
    df = df.loc[~df.token.isna(), ]
    if corpus == 'ontonotes':
        df.rename(columns={'doc_id': 'sentence_id'}, inplace=True)
    if corpus == 'wikiann':
        df.rename(columns={'sentence': 'sentence_id'}, inplace=True)
        df = df.sort_values(['sentence_id', 'token_id'])

    # ensure consistent order of sentences
    df.sentence_id = df.sentence_id.astype(str).str.zfill(6)

    start_validation = timer()
    print('Annotating...', corpus)
        
    sentences = df.groupby('sentence_id')['token'].agg(list).tolist()
    annotations = []
    with tqdm(total=len(sentences), unit='sentence') as pbar:
        for sentence in sentences:
            doc = nlp(" ".join(sentence))
            iob = [token.ent_iob_ if token.ent_iob_ == 'O' else token.ent_iob_ + '-' + token.ent_type_ for token in doc]
            assert len(iob) == len(sentence), f'Error! length of Spacy output does not match: {sentence}.'
            annotations += iob
            pbar.update(1)

    end_validation = timer()
    validation_time = timedelta(seconds=end_validation-start_validation)

    df['spacy_ner'] = pd.Series(annotations)

    df['spacy_ner'] = df.spacy_ner.replace(spacy2conll)

    predictions = df.groupby('sentence_id')['spacy_ner'].agg(list).to_list()

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
