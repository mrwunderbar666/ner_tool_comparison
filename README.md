# Data

- CoNLL 2002
- CoNLL 2003
- Europeana
- OntoNotes
- WikiANN
- CNEC 2.0

## Data Conversion Scripts

Collection of scripts that automatically retrieve the datasets (if possible) and then convert them to a common format. Each corpus is in tokenized long format (one row = one token) and contains the following columns:

- `dataset`: name of dataset
- `language`: language of dataset / tokens
- `subset`: Original name of subset (or split) of dataset. E.g., training, validation, etc.
- `sentence_id`: id of sentence (string), typically enumerated from `000001`. In some cases the corpus also has document ids, then the `sentence_id` includes the `doc_id` as well. E.g, `0001_000001`.
- `token_id`: id (actually position) of token within the sentence. Always starts at 1.
- `token`: actual token in its original form.
- `CoNLL_IOB2`: Named entity tag according to *Inside-Outside-Beginning* scheme as defined by CoNLL. Named entities are limited to Persons, Organizations, Location, and Misc. 

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