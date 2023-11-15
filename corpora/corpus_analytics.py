import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path.cwd()))

from utils.downloader import downloader
from utils.registry import load_registry
from utils.metrics import compute_metrics

registry = load_registry()

p = Path.cwd() / 'corpora'
wgnd_destination = p / 'wgnd.csv'

# Raffo, Julio, 2021, "WGND 2.0", https://doi.org/10.7910/DVN/MSEGSJ, Harvard Dataverse, V1
print('Getting WGND 2.0 from harvard dataverse')
wgnd_url = "https://dataverse.harvard.edu/api/access/datafile/4750353"
downloader(wgnd_url, wgnd_destination)

wgnd_dictionary = pd.read_csv(wgnd_destination)

languages = set(registry.language.unique().tolist()) & set(wgnd_dictionary.langcode.unique().tolist())
languages = list(languages)

corpora = registry.corpus.unique().tolist()
results = []
results_destination = p / 'summary.csv'

splits = ['train', 'test', 'validation']

for language in languages:
    print('Language', language)
    wgnd_subset = {name['name']: name['gender'] for name in wgnd_dictionary.loc[wgnd_dictionary.langcode == language, :].to_dict(orient='records')}
    if len(wgnd_subset) == 0:
        print('Dictionary is empty for language', language)
        continue

    for corpus_name in corpora:
        filt_corpus = (registry['corpus'] == corpus_name) & (registry['language'] == language)
        if len(registry[filt_corpus]) == 0:
            continue
        print('Loading corpus', corpus_name)
        for split in splits:
            try:
                corpus_meta = registry.loc[filt_corpus & (registry.split == split)].to_dict(orient='records')[0]
            except IndexError:
                print('corpus', corpus_name, 'does not have a', split, 'split for language', language)
                continue
            corpus = pd.read_feather(corpus_meta['path'])
            ner_tags_summary = corpus.CoNLL_IOB2.value_counts().to_dict()

            corpus['entity_id'] = ""
            filt = corpus.CoNLL_IOB2.str.startswith('B')
            corpus.loc[filt, 'entity_id'] = corpus.loc[filt, 'sentence_id'] + '_' + corpus.loc[filt, 'token_id'].astype(str) + '_' + corpus.loc[filt, 'CoNLL_IOB2'].str.replace('B-', '')

            for i, row in corpus.iterrows():
                if row['CoNLL_IOB2'].startswith('I'):
                    corpus.loc[i, 'entity_id'] = corpus.loc[i-1, 'entity_id']

            persons = corpus.loc[corpus.CoNLL_IOB2.str.contains('B-PER'), ['token', 'CoNLL_IOB2', 'sentence_id', 'entity_id']]
            
            persons['first_name'] = persons.token
            
            # special treatment for chinese
            if 'zh' in language:
                persons['name_len'] = persons.token.str.len()
                persons.loc[persons.name_len == 3, 'first_name'] = persons.loc[persons.name_len == 3, 'token'].apply(lambda x: x[1:])
            
            persons['gender'] = persons.first_name.str.lower().map(wgnd_subset)
            gender_distribution = persons.gender.fillna('NA').value_counts().to_dict()
            missing_females = {}

            if split == 'validation' and gender_distribution.get('F', 0) > 0:

                # calc precision / recall if no female name was found
                female_entities = persons[persons.gender == 'F'].entity_id.to_list()
                male_entities = persons[persons.gender == 'M'].entity_id.to_list()
                filt = corpus.entity_id.isin(female_entities)
                references = corpus.groupby('sentence_id').agg(list)['CoNLL_IOB2'].to_list()
                predictions = corpus.copy()
                predictions.loc[filt, 'CoNLL_IOB2'] = 'O'
                predictions = predictions.groupby('sentence_id').agg(list)['CoNLL_IOB2'].to_list()
                metrics = compute_metrics(predictions, references)
                missing_females = {"ignore_females_" + k: v for k, v in metrics['PER'].items()}
                
                # create female subcorpus
                filt = corpus.entity_id.isin(female_entities)
                filt2 = corpus[filt].sentence_id
                female_corpus = corpus[filt].reset_index(drop=True)
                female_corpus['subset'] = 'validation_female'
                print('Saving female only validation split ...')
                female_corpus.to_feather(corpus_meta['path'] + '.female', compression="uncompressed")

                # create male subcorpus
                filt = corpus.entity_id.isin(male_entities)
                filt2 = corpus[filt].sentence_id
                male_corpus = corpus[filt].reset_index(drop=True)
                male_corpus['subset'] = 'validation_male'
                print('Saving male only validation split ...')
                male_corpus.to_feather(corpus_meta['path'] + '.male', compression="uncompressed")

            corpus_statistics = {**corpus_meta, **ner_tags_summary, **gender_distribution, **missing_females}
            results.append(corpus_statistics)

pd.DataFrame(results).to_csv(results_destination, index=False)