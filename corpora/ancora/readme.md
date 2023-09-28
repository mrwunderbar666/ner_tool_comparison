# AnCora Corpus

- Citation: Taulé, M., M.A. Martí, M. Recasens (2008) 'Ancora: Multilevel Annotated Corpora for Catalan and Spanish', 
   Proceedings of 6th International Conference on Language Resources and Evaluation. Marrakesh (Morocco). https://aclanthology.org/L08-1222/
- Official Website: https://clic.ub.edu/corpus/en/ancora
- Documentation: https://clic.ub.edu/corpus/sites/default/files/inline-files/ancora-corpus.pdf
- Licence: GNU Public License (see Licence file inside `ancora-es-2.0.0_2.zip`). Redistribution granted by GNU Public Licence


# Parse the Data

Run

```bash
python corpora/ancora/parse_ancora.py
```

Does the following:

- unpacks the zip file
- parses the individual XML documents
- generates a training, test, and validation split