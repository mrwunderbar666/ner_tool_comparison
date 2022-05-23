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

language = 'fr'
p = Path.cwd() / 'tools' / 'spacy'
results_path = Path.cwd() / 'results' / f'spacy_{language}.csv'

models = ['fr_core_news_lg']

corpora = {'europeana': Path.cwd() / 'corpora' / 'europeana' / 'enp_FR.bnf_validation.feather',
            'wikiann': Path.cwd() / 'corpora' / 'wikiann' / 'wikiann-fr_validation.feather'
            }

metric = load_metric("seqeval")
evaluations = []


for model in models:

    print('Evaluating model:', model)

    nlp = spacy.load(model, disable=["parser"])
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
        df.sentence_id = df.sentence_id.astype(str).str.zfill(len(str(df.sentence_id.max())))

        if corpus == 'ned.testb':
            df.sentence_id = df.doc + '_' + df.sentence_id
            df = df.sort_values(['sentence_id', 'token_id'])

        # re-arrange corpus into sentences    
        sentences = df.groupby('sentence_id')['token'].agg(list)

        start_validation = timer()
        print('Annotating...', corpus)
        with tqdm(total=len(sentences), unit='sentence') as pbar:
            for i, sentence in sentences.iteritems():
                # join tokens to sentence and pass it into the tool
                doc = nlp(" ".join(sentence))
                # restore a list of tokens with annotations
                iob = [token.ent_iob_ if token.ent_iob_ == 'O' else token.ent_iob_ + '-' + token.ent_type_ for token in doc]
                assert len(iob) == len(sentence), f'Error! length of Spacy output does not match: {sentence}.'
                sentences[i] = iob
                pbar.update(1)

        end_validation = timer()
        validation_time = timedelta(seconds=end_validation-start_validation)
        assert all(sentences.explode().index == df.sentence_id), 'IDs of annotations and dataframe do not match!'

        # rejoin annotations with dataframe
        df['spacy_ner'] = sentences.explode().values

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
        r['model'] = model
        r['language'] = language
        r['corpus'] = corpus
        r['validation_duration'] = validation_time.total_seconds()

        evaluations.append(r)

results_df = pd.concat(evaluations)
results_df.to_csv(results_path, index=False)
