# NLTK

- Website: https://www.nltk.org/
- Algorithm: Maximum entropy
- Inner workings of maximum entropy classifier: http://mattshomepage.com/articles/2016/May/23/nltk_nec/
- Example for training custom classifier: https://github.com/arop/ner-re-pt/wiki/NLTK
- Spanish Tagger: https://github.com/alvations/spaghetti-tagger

## Training Material

No clear description, but name of NER model indicates that it is trained with the ACE corpus

> 'ACE Named Entity Chunker (Maximum entropy)'

# Instructions


Install with

```bash
python tools/nltk/get_dependencies.py
```

Then you can run the evaluation with

```bash
python tools/nltk/english.py
```

And test the challenges with

```bash
python tools/nltk/challenges.py
```
