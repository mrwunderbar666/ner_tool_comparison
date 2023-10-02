# Installation of Requirements

Make sure to install all required packages (Python & R) before proceeding.

1. Create a virtual environment
    - python3 -m venv .venv
    - source .venv/bin/activate
2. Execute the script `install_prerequisites.sh`

## Manual Installation

### Install Python Dependencies

```
python -m pip install -r requirements.txt
```

Additionally, get a script from huggingface:

```
curl https://huggingface.co/datasets/conll2003/raw/main/conll2003.py -o utils/conll2003.py
```

Then, get spaCy models

```
python -m spacy download zh_core_web_lg
python -m spacy download zh_core_web_trf
python -m spacy download nl_core_news_lg
python -m spacy download en_core_web_lg
python -m spacy download fr_core_news_lg
python -m spacy download de_core_news_lg
python -m spacy download es_core_news_lg
python -m spacy download xx_ent_wiki_sm
```

### Install R Packages
```
Rscript r_packages.r
```

### Install Tools

python3 tools/corenlp/get_corenlp.py
Rscript tools/icews/get_icews.r
python3 tools/jrcnames/get_jrc.py
python3 tools/nltk/get_dependencies.py
python3 tools/opennlp/get_opennlp.py


# Data

The datasets for evaluation are the following:

- CoNLL 2002 (Dutch & Spanish)
- CoNLL 2003 (English & German*)
- Europeana (German, French, Dutch)
- GermEval2014 (German)
- WNUT Emerging Entities (English)
- OntoNotes* (English, Chinese, Arabic)
- WikiANN* (many)
- CNEC 2.0 (Czech)

Alomst every dataset can be downloaded automatically with the supplied scripts. The datasets marked with an asterisk (*) require user intervention. Please refer to the `readme.md` files in the corresponding sub-directories for instructions.

**Please be aware that some datasets are very large and take a while to download and convert**

## Data Conversion Scripts

Collection of scripts that automatically retrieve the datasets (if possible) and then convert them to a common format. 

**Every script should be run from the root directory:** For example, if you want to automatically get the CoNLL2002 dataset run the following `python corpora/conll/get_conll2002.py`

When you run the scripts that automatically download and convert the corpora, a `registry.csv` is created that contains meta-information on each corpus. This file is used by the evaluation scripts to automatically find all available datasets and run the tests.

Each corpus is in tokenized long format (one row = one token) and contains the following columns:

- `dataset`: name of dataset
- `language`: language of dataset / tokens
- `subset`: Original name of subset (or split) of dataset. E.g., training, validation, etc.
- `sentence_id`: id of sentence (string), typically enumerated from `000001`. In some cases the corpus also has document ids, then the `sentence_id` includes the `doc_id` as well. E.g, `0001_000001`.
- `token_id`: id (actually position) of token within the sentence. Always starts at 1.
- `token`: actual token in its original form.
- `CoNLL_IOB2`: Named entity tag according to *Inside-Outside-Beginning* scheme as defined by CoNLL. Named entities are limited to Persons, Organizations, Location, and Misc. 

# NER Tools

- CoreNLP
- NLTK
- ICEWS
- JRC Names
- Nametagger
- OpenNLP
- spaCy
- XLM-RoBERTa (via Huggingface)

## Automatically Getting & Installing Tools

**Every script should be run from the root directory:** For example, if you want to automatically get the CoreNLP  run the following `python tools/corenlp/get_corenlp.py`

# Other Tools

- https://sites.google.com/site/rmyeid/projects/polylgot-ner
- Stanza
- Flair
- NERF (Polish): http://nkjp.pl/index.php?page=14&lang=1

# More Corpora

## English

- [Ultra-Fine Entity Typing (ACL 2018)](https://www.cs.utexas.edu/~eunsol/html_pages/open_entity.html) (Open-ended entity recognition)
- [Few-NERD](https://ningding97.github.io/fewnerd/)

## French

License for Quaero corpus prohibits to train a model with the data and to redistribute the resulting model. Hence corpus only for validation purposes.

- [Quaero Broadcast News Extended Named Entity corpus Corpus](http://catalog.elra.info/en-us/repository/browse/ELRA-S0349/)
- [Quaero Old Press Extended Named Entity corpus Corpus](http://catalog.elra.info/en-us/repository/browse/ELRA-W0073/)

## Dutch

- [SoNaR-1](https://taalmaterialen.ivdnt.org/download/tstc-sonar-corpus/): 1-Million Words subset is manually annotated. 
    - https://taalmaterialen.ivdnt.org/wp-content/uploads/documentatie/sonar_verschillendecorpora.txt
    - https://taalmaterialen.ivdnt.org/wp-content/uploads/documentatie/sonar_documentatie.pdf

## Polish

- [NJKP](http://nkjp.pl/index.php?page=14&lang=1): 1-million-word subcorpus. The manually annotated 1-million word subcorpus of the NJKP, available on CC-BY 4.0.

## Russian

- bsnlp-2019: http://bsnlp.cs.helsinki.fi/bsnlp-2019/shared_task.html (Russian, Czech, Polish, Bulgarian)
- https://www.dialog-21.ru/evaluation/2016/ner/
- NERUS: https://github.com/natasha/nerus

## Hungarian

- "Hungarian Named Entity Corpora": György Szarvas, Richárd Farkas, László Felföldi, András Kocsor, János Csirik: Highly accurate Named Entity corpus for Hungarian. International Conference on Language Resources and Evaluation 2006, Genova (Italy).
    - Website: https://rgai.inf.u-szeged.hu/node/130 (download link broken)
    - Paper: http://www.inf.u-szeged.hu/projectdirs/hlt/papers/lrec_ne-corpus.pdf
- NYTK-NerKor: Simon, Eszter; Vadász, Noémi. (2021) Introducing NYTK-NerKor, A Gold Standard Hungarian Named Entity Annotated Corpus. In: Ekštein K., Pártl F., Konopík M. (eds) Text, Speech, and Dialogue. TSD 2021. Lecture Notes in Computer Science, vol 12848. Springer, Cham. https://doi.org/10.1007/978-3-030-83527-9_19
    - Repository: https://github.com/nytud/NYTK-NerKor

## Japanese

- Megagon Labs Tokyo NE Extension: https://github.com/megagonlabs/UD_Japanese-GSD

## Italian

- Italian Content Annotation Bank (I-CAB) https://ontotext.fbk.eu/icab.html
    - Foundation for other tasks
    - Requires licence agreement
- EVALITA 2011: Named Entity Recognition on Transcribed Broadcast News (https://www.evalita.it/campaigns/evalita-2011/tasks/named-entities/)
- EVALITA 2016: Named Entity rEcognition and Linking in Italian Tweets Task (http://neel-it.github.io/)
    - Data shared in protected GDrive
- KIND (Kessler Italian Named-entities Dataset)
    - Part of EVALITA 2023
    - Repository: https://github.com/dhfbk/KIND/tree/main
    - Permissive licence



## Collections of more corpora (other domains)

- https://github.com/juand-r/entity-recognition-datasets
- https://github.com/davidsbatista/NER-datasets

# Difficult Examples

See the file `challenges.json` for a set of sentences which pose challenges for NER tools.
