# AQMAR

- Website: http://www.cs.cmu.edu/~ark/ArabicNER/
- Citation: Behrang Mohit, Nathan Schneider, Rishav Bhowmick, Kemal Oflazer, and Noah A. Smith. 2012. Recall-Oriented Learning of Named Entities in Arabic Wikipedia. In Proceedings of the 13th Conference of the European Chapter of the Association for Computational Linguistics, pages 162–173, Avignon, France. Association for Computational Linguistics. https://aclanthology.org/E12-1017/
- License: dataset is released under the Creative Commons Attribution-ShareAlike 3.0 Unported license. Redistribution is granted by license

## Text Materials

> This is a 74,000-token corpus of 28 Arabic Wikipedia articles hand-annotated for named entities.



# Parse the Data

Run

```bash
python corpora/aqmar/parse_aqmar.py
```

Does the following:

- unpacks the zip file
- parses the individual TXT documents
- applies minor fixes
- generates a training, test, and validation split


# Acknowledgements

As stated on website:

> This research was supported by Qatar National Research Fund grant NPRP 08-485-1-083.