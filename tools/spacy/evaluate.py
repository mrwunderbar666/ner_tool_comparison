import sys
import re
import spacy
spacy.prefer_gpu()
from spacy.tokenizer import Tokenizer
from pathlib import Path
import pandas as pd
from timeit import default_timer as timer
from datetime import timedelta

from datasets import load_metric

from tqdm import tqdm

sys.path.insert(0, str(Path.cwd()))
from utils.mappings import spacy2conll
from utils.registry import load_registry

languages = ['en', 'nl', 'fr', 'de', 'es', 'zh']
p = Path.cwd() / 'tools' / 'spacy'

results_path = Path.cwd() / 'results' / f'spacy.csv'

models = {'en': ['en_core_web_lg', "en_core_web_trf"],
            'zh': ['zh_core_web_lg', 'zh_core_web_trf'],
            'nl': ['nl_core_news_lg'],
            'fr': ['fr_core_news_lg'],
            'de': ['de_core_news_lg', 'de_dep_news_trf'],
            'es': ['es_core_news_lg'],
            'multi': ['xx_ent_wiki_sm']}

registry = load_registry()
metric = load_metric("seqeval")
evaluations = []


for language in languages:

    print('Evluating:', language)

    if language == 'multi':
        corpora = registry.loc[(registry.language.isin(languages)) & (registry.split == 'validation')]
    else:
        corpora = registry.loc[(registry.language == language) & (registry.split == 'validation')]

    for model in models[language]:

        print('model:', model)

        nlp = spacy.load(model, disable=["parser"])
        # custom tokenizer
        # conll datasets do not split hyphenated words
        nlp.tokenizer = Tokenizer(nlp.vocab, token_match=re.compile(r'\S+').match)


        for _, row in corpora.iterrows():

            path_corpus = Path(row['path'])

            if not path_corpus.exists():
                print('could not find corpus:', path_corpus)
                continue

            print('Loading corpus:', path_corpus)

            df = pd.read_feather(path_corpus)
            df = df.loc[~df.token.isna(), ]

            # re-arrange corpus into sentences    
            sentences = df.groupby('sentence_id')['token'].agg(list)

            start_validation = timer()
            print('Annotating...', path_corpus)
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
