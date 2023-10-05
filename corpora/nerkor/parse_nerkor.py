import sys

import zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
from tqdm import tqdm

sys.path.insert(0, str(Path.cwd()))
from utils.registry import add_corpus
from utils.parsers import parse_conllup


def parse_document(path: Path) -> pd.DataFrame:
    doc = parse_conllup(path)
    doc['document_id'] = path.name.replace('.conllup', '')
    doc['genre'] = path.parent.name
    doc['split'] = path.parent.parent.name
    return doc


if __name__ == '__main__':

    p = Path.cwd() / 'corpora' / 'nerkor'

    tmp = p / 'tmp'

    if not tmp.exists():
        tmp.mkdir(parents=True)

    z = zipfile.ZipFile(p / "nytk-nerkor.zip", mode='r')
    z.extractall(path=tmp)

    conllup_files = list(p.glob('tmp/*/*/*.conllup'))

    parsed_docs = []

    with tqdm(total=len(conllup_files)) as pbar:
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(parse_document, conllup): conllup for conllup in conllup_files}
            for future in as_completed(futures):
                conllup = futures[future]
                try:
                    parsed_docs.append(future.result())
                    pbar.update(1)
                except Exception as e:
                    print(f'{conllup.name} generated an exception:', e)


    corpus = pd.concat(parsed_docs, ignore_index=True).reset_index(drop=True)

    corpus['split'] = corpus['split'].replace({'devel': 'test', 'test': 'validation'})
    corpus['language'] = 'hu'

    corpus['sentence_id'] = corpus['document_id'] + '_' + corpus['sentence_id'].astype(str).str.zfill(4)

    corpus.columns = corpus.columns.str.lower().str.replace(':', '_')

    # fix one typo in dataset
    corpus.loc[corpus.conll_ner == 'V', 'conll_ner'] = 'O' 

    corpus = corpus.rename(columns={'conll_ner': 'CoNLL_IOB2', 'form': 'token'})

    for split in corpus.split.unique().tolist():
        print('Exporting split:', split)
        subset = corpus.loc[corpus.split == split].reset_index(drop=True)

        destination = p / f'nerkor_{split}.feather'

        subset.to_feather(destination, compression='uncompressed')

        print(f"processed {split} and saved to", destination)

        details = {'corpus': 'nerkor', 
                    'subset': f'nerkor', 
                    'path': destination, 
                    'split': split,
                    'language': 'hu', 
                    'tokens': len(subset), 
                    'sentences': len(subset.sentence_id.unique())}

        add_corpus(details)

    print('Done!')