# SoNaR-1

- Website: https://taalmaterialen.ivdnt.org/download/tstc-sonar-corpus/
- Citation: 
    - Documentation: Nelleke Oostdijk, Martin Reynaert, Véronique Hoste, Henk van den Heuvel (2013). SoNaR User Documentation. https://taalmaterialen.ivdnt.org/wp-content/uploads/documentatie/sonar_documentatie.pdf
- Licence: Free for Academic Use; Registration Required

## Text Types

Comprised of a large variety of texts:

- Books
- Brochures
- E-magazines
- E-Newsletters
- Newsletters
- Newspapers
- Legal texts
- Magazines
- Policy documents
- Newspapers
- Autocues
- Minutes
- Wikipedia
- Proceedings
- Reports
- Speeches
- Teletext

# How To Retrieve

1. Go to the [official website](https://taalmaterialen.ivdnt.org/download/tstc-sonar-corpus/) and register there. You may need to wait a few days until your account gets activated
2. Download the very large corpus file (53 GB)
3. We only need to extract a subset of files:
    - `tar xf sonar.tgz --occurrence ./SoNaRCorpus_NC_1.2/SONAR1/NE/SONAR_1_NE/IOB`
    - This command assumes that the downloaded file is named "sonar.tgz" and it will extract only the named entity annotation files
4. copy the extracted `IOB` folder to this repository under `corpora/sonar/tmp/`
5. Run `python3 corpora/sonar/parse_sonar.py`
    - Parses each IOB file
    - Joins into one large corpus
    - applies mapping
    - Then splits into train, validation and test set