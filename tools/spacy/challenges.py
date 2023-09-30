import sys
import spacy
spacy.prefer_gpu()
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path.cwd()))
from utils.challenges import load_challenges

p = Path.cwd() / 'tools' / 'spacy'

results_path = Path.cwd() / 'results' / f'spacy_challenges.json'

models = {'en': "en_core_web_trf",
          'zh': 'zh_core_web_trf',
          'nl': 'nl_core_news_lg',
          'fr': 'fr_core_news_lg',
          'de': 'de_core_news_lg',
          'es': 'es_core_news_lg'
          }


challenges = load_challenges()

challenges['tool'] = 'spacy'
challenges['tokens'] = ''
challenges['iob'] = ''

for language, model in models.items():

    if language not in challenges.language.unique():
        continue

    print('Evaluating challenges for:', language)

    challenge_sentences = challenges.loc[(challenges.language == language)]

    nlp = spacy.load(model, disable=["parser"])
    # custom tokenizer
    # conll datasets do not split hyphenated words

    for index, row in challenge_sentences.iterrows():

        doc = nlp(row['text'])
        # restore a list of tokens with annotations
        tokens = [token.text for token in doc]
        iob = [token.ent_iob_ if token.ent_iob_ == 'O' else token.ent_iob_ + '-' + token.ent_type_ for token in doc]
        
        challenges.at[index, 'tokens'] = tokens
        challenges.at[index, 'iob'] = iob


challenges.to_json(results_path, orient="records")

