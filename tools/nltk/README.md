# Algorithm

Maximum entropy

# Documentation

How to: https://www.nltk.org/book/ch07.html

NLTK provides a classifier that has already been trained to recognize named entities, accessed with the function nltk.ne_chunk(). If we set the parameter binary=True [1], then named entities are just tagged as NE; otherwise, the classifier adds category labels such as PERSON, ORGANIZATION, and GPE.

# Resources

- Inner workings of maximum entropy classifier: http://mattshomepage.com/articles/2016/May/23/nltk_nec/
- Example for training custom classifier: https://github.com/arop/ner-re-pt/wiki/NLTK
- Spanish Tagger: https://github.com/alvations/spaghetti-tagger
