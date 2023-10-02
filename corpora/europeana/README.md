# Europeana

- Citation: Neudecker, C. (2016). An Open Corpus for Named Entity Recognition in Historic Newspapers. In N. Calzolari, K. Choukri, T. Declerck, S. Goggi, M. Grobelnik, B. Maegaard, J. Mariani, H. Mazo, A. Moreno, J. Odijk & S. Piperidis (Eds.), Proceedings of the Tenth International Conference on Language Resources and Evaluation (LREC 2016). European Language Resources Association (ELRA). https://aclanthology.org/L16-1689/
- Repository: https://github.com/EuropeanaNewspapers/ner-corpora

# Issues

## enp_FR.bnf

- Uses different annotation scheme (see: https://github.com/EuropeanaNewspapers/ner-corpora/issues/44)

## enp_DE.sbb

- many problems, requires separate patch file to manually fix data structure

## enp_de.lft

- one tag is `B-BER`, but should be `B-PER`

# Get the Data

Run

```bash
python corpora/europeana/get_europeana.py
```

Above script automatically downloads, parses, patches, and converts the corpora.
