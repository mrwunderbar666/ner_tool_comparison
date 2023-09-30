import sys
from pathlib import Path
import pandas as pd
import nltk

sys.path.insert(0, str(Path.cwd()))
from utils.challenges import load_challenges

language = 'en'
p = Path.cwd() / 'tools' / 'nltk'
results_path = Path.cwd() / 'results' / 'nltk_challenges.json'

challenges = load_challenges()

challenges['tool'] = 'nltk'
challenges['tokens'] = ''
challenges['iob'] = ''

challenge_sentences = challenges.loc[(challenges.language == language)]

print('Evaluating:', language)

tokenizer = nltk.tokenize.NLTKWordTokenizer()

for index, row in challenge_sentences.iterrows():
    
    tokens = tokenizer.tokenize(row['text'])
    pos = nltk.pos_tag(tokens)
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
    challenges.at[index, 'tokens'] = tokens
    challenges.at[index, 'iob'] = tagged

challenges.to_json(results_path, orient="records")

print('Done!')