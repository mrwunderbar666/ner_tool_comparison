import sys
from pathlib import Path
import pandas as pd
from tqdm import tqdm 
import zipfile
import tarfile
import os 
import gc
p = Path.cwd() / 'corpora' / 'wikiann'
tmp = p / 'tmp'

if not tmp.exists():
    tmp.mkdir()

# z_files = list(p.glob('*.zip'))

# assert len(z_files) > 0, 'Could not find zip file!'


# z = zipfile.ZipFile(z_files[0], mode='r')
# z.extractall(path=tmp)

# gz_files = list(tmp.glob('name_tagging/*.gz'))

# with tqdm(total=len(gz_files), unit='tar.gz') as pbar:
#     for gz in gz_files:
#         with tarfile.open(gz, 'r:gz', errorlevel=1) as tar:
#             for f in tar:
#                 if f.name.lower() == 'readme.txt': continue
#                 try:
#                     tar.extract(f, path=tmp)
#                 except IOError as e:
#                     os.remove(tmp / f.name)
#                     tar.extract(f, path=tmp)
#                 finally:
#                     os.chmod(tmp / f.name, f.mode)
#         pbar.update(1)


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
                            'sentence': sentence,
                            'token_id': token_id,
                            'token': l[0],
                            'IOB2': l[-1]})
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
        df = pd.concat(tmp_dfs)
        del tmp_dfs
        df = df.sort_values(by=['sentence', 'token'])
        df = df.reset_index(drop=True)
        df.to_feather(p / bio.name.replace('.bio', '.feather'), compression='uncompressed')
        print('success!')
        for tmp_f in tmp_files:
            os.remove(tmp_f)

    elif len(output) > 0:
        print('storing dataset with', len(output), 'tokens')
        df = pd.DataFrame(output)
        df.reset_index(drop=True, inplace=True)
        df.to_feather(p / bio.name.replace('.bio', '.feather'), compression='uncompressed')

for raw in bio_files:
    print('Parsing:', raw)
    parse_bio(raw)


# from concurrent.futures import ThreadPoolExecutor, as_completed



# with tqdm(total=len(results['files']), unit='file') as pbar:
#     with ThreadPoolExecutor(max_workers=4) as executor:
#         futures = {executor.submit(download_gdrive_file, gfile, tmp): gfile for gfile in results['files']}
#         for future in as_completed(futures):
#             print(future.result())
#             pbar.update(1)
