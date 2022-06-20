import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm 
import zipfile
import tarfile
import os 
import gc
import shutil
from sklearn.model_selection import train_test_split
sys.path.insert(0, str(Path.cwd()))
from utils.registry import add_corpus

seed = 5618

p = Path.cwd() / 'corpora' / 'wikiann'
tmp = p / 'tmp'

if not tmp.exists():
    tmp.mkdir()

z_files = list(p.glob('*.zip'))

assert len(z_files) > 0, 'Could not find zip file!'

print('Extracting zip archive')
z = zipfile.ZipFile(z_files[0], mode='r')
z.extractall(path=tmp)

gz_files = list(tmp.glob('name_tagging/*.gz'))

print(f'found {len(gz_files)} corpus files. Deflating...')
with tqdm(total=len(gz_files), unit='tar.gz') as pbar:
    for gz in gz_files:
        with tarfile.open(gz, 'r:gz', errorlevel=1) as tar:
            for f in tar:
                if f.name.lower() == 'readme.txt': continue
                try:
                    tar.extract(f, path=tmp)
                except IOError as e:
                    os.remove(tmp / f.name)
                    tar.extract(f, path=tmp)
                finally:
                    os.chmod(tmp / f.name, f.mode)
        pbar.update(1)

bio_files = list(tmp.glob('*.bio'))

def parse_bio(bio):
    language = bio.name.replace('wikiann-', '').replace('.bio', '')
    output = []
    sentence = 1
    token_id = 1
    part = 0
    with open(bio) as f:
        for l in f:
            if l.strip() == '':
                sentence += 1
                token_id = 1
                continue
            l = l.split()
            output.append({'corpus': 'wikiann',
                            'language': language,
                            'sentence_id': str(sentence).zfill(6),
                            'token_id': token_id,
                            'token': l[0],
                            'CoNLL_IOB2': l[-1]})
            token_id += 1
            if len(output) > 1e7:
                print('part:', part)
                tmp_df = pd.DataFrame(output)
                tmp_df.reset_index(drop=True, inplace=True)
                tmp_df.to_feather(p / "tmp" / f"{bio.name.replace('.bio', '.feather')}_part{part}", compression='uncompressed')
                part += 1
                del tmp_df
                output = []
                gc.collect()

    if part > 0:
        print("Rejoining partitioned datafile...")
        tmp_files = list(p.glob(f"tmp/{bio.name.replace('.bio', '.feather')}*"))
        tmp_dfs = [pd.read_feather(tmp_f) for tmp_f in tmp_files]
        tmp_dfs.append(pd.DataFrame(output))
        df = pd.concat(tmp_dfs)
        del tmp_dfs
        df = df.sort_values(by=['sentence_id', 'token_id'])
        print('success!')
        for tmp_f in tmp_files:
            os.remove(tmp_f)

    elif len(output) > 0:
        print('storing dataset with', len(output), 'tokens')
        df = pd.DataFrame(output)
        

    if len(output) > 0:
        df.reset_index(drop=True, inplace=True)
        # split into train (70%), test (15%), validate (15%)
        # BUT maximum number of test / validation sample of ~100k sentences
        sentence_index = df.sentence_id.unique().tolist()
        if len(sentence_index) > 660000:
            train, test_val = train_test_split(sentence_index, test_size=200000, random_state=seed)
        else:
            train, test_val = train_test_split(sentence_index, test_size=0.3, random_state=seed)
        test, val = train_test_split(test_val, test_size=0.5, random_state=seed)
        df_train = df.loc[df.sentence_id.isin(train), ]
        df_test = df.loc[df.sentence_id.isin(test), ]
        df_val = df.loc[df.sentence_id.isin(val), ]

        df_train.reset_index(inplace=True, drop=True)
        df_test.reset_index(inplace=True, drop=True)
        df_val.reset_index(inplace=True, drop=True)

        train_destination = p / bio.name.replace('.bio', '_train.feather')
        test_destination = p / bio.name.replace('.bio', '_test.feather')
        val_destination = p / bio.name.replace('.bio', '_validation.feather')
        corpus_details = {'corpus': 'wikiann',
                      'subset': bio.name,
                      'path': train_destination, 
                      'split': 'train',
                      'language': language, 
                      'tokens': len(df_train), 
                      'sentences': len(df_train.sentence_id.unique())}
        add_corpus(corpus_details)

        corpus_details['path'] = test_destination
        corpus_details['split'] = 'test'
        corpus_details['tokens'] = len(df_test)
        corpus_details['sentences'] = len(df_test.sentence_id.unique())
        add_corpus(corpus_details)

        corpus_details['path'] = val_destination
        corpus_details['split'] = 'validation'
        corpus_details['tokens'] = len(df_val)
        corpus_details['sentences'] = len(df_val.sentence_id.unique())
        add_corpus(corpus_details)

        df_train.sort_values(['sentence_id', 'token_id'])
        df_train.to_feather(train_destination, compression='uncompressed')
        df_test.sort_values(['sentence_id', 'token_id'])
        df_test.to_feather(test_destination, compression='uncompressed')
        df_val.sort_values(['sentence_id', 'token_id'])
        df_val.to_feather(val_destination, compression='uncompressed')


err = []

for raw in bio_files:
    print('Parsing:', raw)
    try:
        parse_bio(raw)
    except Exception as e:
        print('parsing failed!', e)
        err.append((raw, e))


shutil.rmtree(tmp)

print('Done!')

if len(err) > 0:
    print('Following datasets could not be parsed successfully:')
    for e in err:
        print('Dataset:', e[0])
        print('Error:', e[1])