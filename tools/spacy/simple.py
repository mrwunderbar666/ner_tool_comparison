import spacy
import re
nlp = spacy.load("en_core_web_sm", disable=["parser"])
doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)

ent = [doc[0].text, doc[0].ent_iob_, doc[0].ent_type_]

from helpers import infix_re

nlp.tokenizer.infix_finditer = infix_re.finditer

nlp.tokenizer = Tokenizer(nlp.vocab, token_match=re.compile(r'\S+').match)


import pandas as pd
from pathlib import Path

df = pd.read_feather(Path.cwd() / 'corpora' / 'conll' / 'conll2003_en_validation_iob.feather')
df = df[~df.token.isna()]
df.sentence_id = df.sentence_id.astype(str).str.zfill(4)

sentences = df.groupby('sentence_id')['token'].agg(list).tolist()

doc = nlp(" ".join(sentences[2]))

iob = []

iob = [token.ent_iob_ if token.ent_iob_ == 'O' else token.ent_iob_ + '-' + token.ent_type_ for token in doc ]

for token in doc:
    print(token.text, token.ent_iob_, token.ent_type_)
    if token.ent_iob_ == 'O':
        iob.append(token.ent_iob_)
    else:
        iob.append(token.ent_iob_ + '-' + token.ent_type_)

transformer = spacy.load("en_core_web_trf")

doc = transformer(" ".join(sentences[2]))

iob = []

for token in doc:
    print(token.text, token.ent_iob_, token.ent_type_)
    if token.ent_iob_ == 'O':
        iob.append(token.ent_iob_)
    else:
        iob.append(token.ent_iob_ + '-' + token.ent_type_)
