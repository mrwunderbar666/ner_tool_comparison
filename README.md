# Installation of Requirements

Make sure to install all required packages (Python & R) before proceeding.

## Python

```
python -m pip install -r requirements.txt
```

# R
```
Rscript r_packages.r
```

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

# More Corpora

## English

- [Ultra-Fine Entity Typing (ACL 2018)](https://www.cs.utexas.edu/~eunsol/html_pages/open_entity.html)
- [Few-NERD](https://ningding97.github.io/fewnerd/)

## Portuguese

- [HAREM](https://www.linguateca.pt/HAREM/)

## French

License for Quaero corpus prohibits to train a model with the data and to redistribute the resulting model. Hence corpus only for validation purposes.

- [Quaero Broadcast News Extended Named Entity corpus Corpus](http://catalog.elra.info/en-us/repository/browse/ELRA-S0349/)
- [Quaero Old Press Extended Named Entity corpus Corpus](http://catalog.elra.info/en-us/repository/browse/ELRA-W0073/)

## Spanish

- [AnCora](http://clic.ub.edu/corpus/en/ancora) (Catalan & Spanish)

## Dutch

- [SoNaR-1](https://taalmaterialen.ivdnt.org/download/tstc-sonar-corpus/). 
    - https://taalmaterialen.ivdnt.org/wp-content/uploads/documentatie/sonar_verschillendecorpora.txt
    - https://taalmaterialen.ivdnt.org/wp-content/uploads/documentatie/sonar_documentatie.pdf

# Difficult Examples

See the file `challenges.json` for a set of sentences which pose challenges for NER tools.