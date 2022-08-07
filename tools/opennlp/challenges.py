import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path.cwd()))
from tools.opennlp.opennlp import annotate

languages = ['en', 'nl', 'es']

p = Path.cwd() / 'tools' / 'opennlp'
opennlp_dir = list(p.glob('apache-opennlp-*'))[0]
opennlp_bin = opennlp_dir / 'bin' / 'opennlp'

results_path = Path.cwd() / 'results' / f'opennlp_challenges.json'

challenges = pd.read_json(Path.cwd() / 'challenges.json')

challenges['tool'] = 'opennlp'
challenges['tokens'] = ''
challenges['iob'] = ''

for language in languages:

    if language not in challenges.language.unique():
        continue

    print('Evaluating:', language)
    models = {'person': p / 'models' / f'{language}-ner-person.bin',
            'organization': p / 'models' / f'{language}-ner-organization.bin',
            'location': p / 'models' / f'{language}-ner-location.bin',
            'misc': p / 'models' / f'{language}-ner-misc.bin'}

    challenge_sentences = challenges.loc[(challenges.language == language)]

    for index, row in challenge_sentences.iterrows():

        df = pd.DataFrame()

        for model, model_path in models.items():
            if not model_path.exists(): continue
            tagged = annotate(row['text'], opennlp_bin, model=model_path)
            tagged = pd.Series(tagged).explode().reset_index(drop=True)
            df[model] = tagged

        df['opennlp_ner'] = 'O'

        for model in models.keys():
            if model not in df.columns: continue
            filt = df['opennlp_ner'] == 'O'
            df.loc[filt, 'opennlp_ner'] = df.loc[filt, model]

        challenges.at[index, 'iob'] = df.opennlp_ner.to_list()


challenges.to_json(results_path, orient="records")

print('Done!')