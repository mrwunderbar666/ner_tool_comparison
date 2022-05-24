import nltk

sentence = "Barack Obama went to Vienna on Friday to meet Alexei Navalny."

tokens = nltk.word_tokenize(sentence)

# Resource punkt not found.
#   Please use the NLTK Downloader to obtain the resource:

#   >>> import nltk
#   >>> nltk.download('punkt')

pos = nltk.pos_tag(tokens)

# Resource averaged_perceptron_tagger not found.
#   Please use the NLTK Downloader to obtain the resource:

#   >>> import nltk
#   >>> nltk.download('averaged_perceptron_tagger')
  
#   For more information see: https://www.nltk.org/data.html

ne = nltk.ne_chunk(pos)

#   Resource maxent_ne_chunker not found.
#   Please use the NLTK Downloader to obtain the resource:

#   >>> import nltk
#   >>> nltk.download('maxent_ne_chunker')

# ACE Named Entity Chunker (Maximum entropy) [ download | source ]
# id: maxent_ne_chunker; size: 13404747; author: ; copyright: ; license: ;

# Resource words not found.
#   Please use the NLTK Downloader to obtain the resource:

#   >>> import nltk
#   >>> nltk.download('words')

import pandas as pd
from pathlib import Path

df = pd.read_feather(Path.cwd() / 'corpora' / 'conll' / 'conll2003_en_validation_iob.feather')
df = df[~df.token.isna()]
df.sentence_id = df.sentence_id.astype(str).str.zfill(4)

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

nltk2conll = {'B-GPE': 'B-LOC', 'I-GPE': 'I-LOC', 
              'B-GSP': 'B-LOC', 'I-GSP': 'I-LOC', 
              'B-FACILITY': 'B-LOC', 'I-FACILITY': 'I-LOC', 
              'B-LOCATION': 'B-LOC', 'I-LOCATION': 'I-LOC', 
              'B-PERSON': 'B-PER', 'I-PERSON': 'I-PER',
              'B-ORGANIZATION': 'B-ORG', 'I-ORGANIZATION': 'I-ORG', 
              'B-PERSON': 'B-PER', 'I-PERSON': 'I-PER'}

from datasets import load_metric

metric = load_metric("seqeval")
evaluations = []


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
