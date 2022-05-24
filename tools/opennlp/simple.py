from subprocess import Popen, PIPE
from pathlib import Path

from tools.opennlp.utils import annotate, apache2conll

import pandas as pd

language = 'en'
p = Path.cwd() / 'tools' / 'opennlp'
opennlp_dir = list(p.glob('apache-opennlp-*'))[0]
opennlp_bin = opennlp_dir / 'bin' / 'opennlp'
models = list(p.glob(f'models/{language}-*'))


df = pd.read_feather(Path.cwd() / 'corpora' / 'conll' / 'conll2003_en_validation_iob.feather')
df = df.loc[~df.token.isna(), ]
df.sentence_id = df.sentence_id.astype(str).str.zfill(6)
sentences = df.groupby('sentence_id')['token'].agg(list).tolist()

tagged = annotate(" ".join(sentences[0]), opennlp_bin, model=models[-1])

sentences = "\n".join([" ".join(s) for s in sentences])


tagged = annotate(sentences, opennlp_bin, model=models[-1])

p = Popen([opennlp_bin, "TokenNameFinder", model_path],
                shell=False, stdin=PIPE, stdout=PIPE, stderr=PIPE, universal_newlines=True)

(stdout, stderr) = p.communicate(sentences)

annotated = []

for tagged in stdout.split('\n'):
    ner_tag = 'O'
    tags = []
    first = False
    for token in tagged.split(' '):
        if token.startswith('<START:'):
            ner_tag = token.replace('<START:', '').replace('>', '')
            first = True
            continue
        elif token == '<END>':
            ner_tag = 'O'
            continue
        if first:
            tags.append('B-' + ner_tag)
            first = False
        elif ner_tag != 'O':
            tags.append('I-' + ner_tag)
        else:
            tags.append(ner_tag)
    annotated.append(tags)                
