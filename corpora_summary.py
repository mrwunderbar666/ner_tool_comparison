from pathlib import Path
import pandas as pd

p = Path.cwd()

output = p / 'corpora' / 'summary.csv'

corpora = list(p.glob('corpora/*/*.feather'))

infos = []

for c in corpora:

    print('Loading', c)

    df = pd.read_feather(c)

    summary = {'path': str(c), 'file': str(c.name)}

    try:
        summary['dataset'] = df['dataset'][0]
    except:
        summary['dataset'] = ''

    try:
        summary['language'] = df['language'][0]
    except:
        summary['language'] = ''

    try:
        summary['corpus'] = df['corpus'][0]
    except:
        summary['corpus'] = ''

    summary['tokens'] = len(df)
    
    try:
        summary['sentences'] = len(df['sentence_id'].unique())
    except:
        summary['sentences'] = 0

    infos.append(summary)

infos = pd.DataFrame(infos)

infos.to_csv(output, index=False)

print('Done!')