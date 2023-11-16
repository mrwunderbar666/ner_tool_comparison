# Notes for Reviewers

This is the entire codebase for replicating the evaluations / benchmarks of the paper. The scripts for retrieving and building the datasets are also included. However, we attached all datasets as well. Therefore, **there is no need to run the various scripts in the `corpora` directory**.

The tools are not packaged here, but the exact steps for downloading and installing them are below (section "Installation of Requirements").

We included the XLM-RoBERTa model that was evaluated in the paper. In our case its ID is `32872`, you can find it under `tools/xlmroberta/models/32872`. You could re-run the entire hyperparameter optimization scripts as well as fine-tuning scripts. But please note that this would take a very long time.  

Overview of Directories:


- analyse_results: Helper Scripts for making plots and tables
- corpora: All datasets / corpora are in here alongside mini documentations and citation information
- plots: Plots as shown in the paper
- results: Raw evaluation results in csv format
- tools: All tools in this directory alongside documentation
- utils: Collection of helpers that are shared in the codebase


# Installation of Requirements

Make sure to install all required packages (Python & R) before proceeding.

Everything is conveniently handled by the script `install_prerequisites.sh`

## Manual Installation

**Every script should be run from the root directory:** For example, if you want to automatically get and install CoreNLP run the following `python tools/corenlp/get_corenlp.py`

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
python3 -m spacy download zh_core_web_lg-3.3.0 --direct
python3 -m spacy download zh_core_web_trf-3.3.0 --direct
python3 -m spacy download nl_core_news_lg-3.3.0 --direct
python3 -m spacy download en_core_web_lg-3.3.0 --direct
python3 -m spacy download en_core_web_trf-3.3.0 --direct
python3 -m spacy download fr_core_news_lg-3.3.0 --direct
python3 -m spacy download de_core_news_lg-3.3.0 --direct
python3 -m spacy download es_core_news_lg-3.3.0 --direct
python3 -m spacy download xx_ent_wiki_sm-3.3.0 --direct
python3 -m spacy download pt_core_news_lg-3.3.0 --direct
python3 -m spacy download fi_core_news_lg-3.3.0 --direct
python3 -m spacy download ca_core_news_lg-3.3.0 --direct
python3 -m spacy download it_core_news_lg-3.3.0 --direct
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

- AnCora (Spanish &  Catalan)
- AQMAR (Arabic)
- CNEC 2.0 (Czech)
- CoNLL 2002 (Dutch & Spanish)
- CoNLL 2003 (English & German*)
- Europeana (German, French, Dutch)
- FiNER (Finnish)
- GermEval2014 (German)
- HIPE (German, English, French)
- KIND (Italian)
- LÂMPADA - Second HAREM Resource Package
- NYTK NerKor (Hungarian)
- OntoNotes* (English, Chinese, Arabic)
- SoNaR* (Dutch)
- WikiANN* (many)
- WNUT Emerging Entities (English)

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

# Difficult Examples

See the file `challenges.json` for a set of sentences which pose challenges for NER tools.