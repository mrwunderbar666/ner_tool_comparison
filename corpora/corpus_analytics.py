import pandas as pd
from pathlib import Path

import sys
sys.path.insert(0, str(Path.cwd()))

from utils.downloader import downloader
from utils.registry import load_registry

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
        filt = (registry['corpus'] == corpus_name) & (registry['language'] == language)
        if len(registry[filt]) == 0:
            continue
        print('Loading corpus', corpus_name)
        for split in splits:
            corpus_meta = registry.loc[filt & (registry.split == split)].to_dict(orient='records')[0]
            corpus = pd.read_feather(corpus_meta['path'])
            ner_tags_summary = corpus.CoNLL_IOB2.value_counts().to_dict()
            persons = corpus.loc[corpus.CoNLL_IOB2.str.contains('B-PER'), ['token', 'CoNLL_IOB2']]
            
            persons['first_name'] = persons.token
            
            # special treatment for chinese
            if 'zh' in language:
                persons['name_len'] = persons.token.str.len()
                persons.loc[persons.name_len == 3, 'first_name'] = persons.loc[persons.name_len == 3, 'token'].apply(lambda x: x[1:])
            
            persons['gender'] = persons.first_name.str.lower().map(wgnd_subset)
            res = persons.gender.fillna('NA').value_counts().to_dict()
            corpus_statistics = {**corpus_meta, **ner_tags_summary, **res}

            results.append(corpus_statistics)


pd.DataFrame(results).to_csv(results_destination, index=False)