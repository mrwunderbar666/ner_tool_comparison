import sys
from pathlib import Path
import zipfile

sys.path.insert(0, str(Path.cwd()))
from utils.registry import add_corpus
from utils.downloader import downloader
from utils.parsers import parse_hipe
from utils.mappings import hipe2conll

p = Path.cwd() / 'corpora' / 'hipe'


if __name__ == "__main__":

    tmp = p / 'tmp'

    if not tmp.exists():
        tmp.mkdir(parents=True)

    z = zipfile.ZipFile(p / "CLEF-HIPE-shared-task-data.zip", mode='r')
    z.extractall(path=tmp)

    for corpus in tmp.glob('v1.4/*/*.tsv'):

        df = parse_hipe(corpus)

        df.columns = df.columns.str.lower().str.replace('-', '_')
        df = df.rename(columns={'nel_lit': 'named_entity_linked'})

        df['CoNLL_IOB2'] = df['ne_coarse_lit'].replace(hipe2conll, regex=False)

        """
            HIPE annotation count demonyms as Persons, these are additionally tagged
            in NE-FINE-LIT as `B-pers.coll` / `I-pers.coll`
        """

        filt = df['ne_fine_lit'].str.contains('pers.coll')
        df.loc[filt, 'CoNLL_IOB2'] = df.loc[filt, 'CoNLL_IOB2'].str.replace('-PER', '-MISC')

        assert len(df.language.unique()), ValueError(f'Corpus contains more than one language: {corpus}')

        language = df.language.unique()[0]

        df['corpus'] = 'hipe'
        df['subset'] = corpus.name.replace('.tsv', '')

        df.sentence_id = df.document_id.str.replace('-', '_') + '_' + df.sentence_id.astype(str)

        cols = ['corpus', 'subset',
                'language', 'sentence_id', 
                'token_id', 
                'token', 'CoNLL_IOB2', 'ne_coarse_lit', 'ne_coarse_meto', 'ne_fine_lit',
                'ne_fine_meto', 'ne_fine_comp', 'ne_nested', 'named_entity_linked',
                'nel_meto', 'misc']

        df = df.loc[:, cols]
        corpus_destination = p / corpus.name.replace('.tsv', '.feather')
        df.to_feather(corpus_destination, compression='uncompressed')

        split = ''
        if "test" in corpus.name:
            split = 'validation'
        elif "dev" in corpus.name:
            split = 'test'
        elif "train" in corpus.name:
            split = "train"

        corpus_details = {'corpus': 'hipe', 
                            'subset': corpus.name.replace('.tsv', ''), 
                            'path': corpus_destination, 
                            'split': split,
                            'language': language, 
                            'tokens': len(df), 
                            'sentences': len(df.sentence_id.unique())}

        add_corpus(corpus_details)
        print(f"Sucess! Saved to {corpus_destination}")


    print('Done!')