from pathlib import Path
import pandas as pd
import random

random.seed(5618)

def find_challenges_file() -> Path:
    p = Path(__file__).parent.parent
    r = p / 'challenges.json'
    if not r.exists():
        raise FileNotFoundError('Could not find challenges.json')
    return r

def random_upper(s: str) -> str:
    l = s.split()
    half = int(len(l) / 2)
    for i in random.sample(range(len(l)), half):
        l[i] = l[i].upper()
    return " ".join(l)

def load_challenges() -> pd.DataFrame:
    """
        Loads the challenges file and applies simple augmentation

        - Challenges in all UPPERCASE
        - in all lowecase
        - In All Title Case
        - RANDOMLY make THEM upper CASE

    """
    df = pd.read_json(find_challenges_file())
    df['applied_augmentation'] = 'None'

    # run some simple augmentation
    df_upper = df.copy()
    df_upper.text = df_upper.text.str.upper()
    df_upper['applied_augmentation'] = 'upper case'

    df_lower = df.copy()
    df_lower.text = df_lower.text.str.lower()
    df_lower['applied_augmentation'] = 'lower case'

    df_title = df.copy()
    df_title.text = df_title.text.str.title()
    df_title['applied_augmentation'] = 'title case'

    df_random_upper = df.copy()
    df_random_upper.text = df_random_upper.text.apply(random_upper)
    df_random_upper['applied_augmentation'] = 'random upper case'

    df = pd.concat([df, df_upper, df_lower, df_title, df_random_upper], ignore_index=True).reset_index(drop=True)

    return df
    


if __name__ == '__main__':
    print(find_challenges_file())

    print(load_challenges())