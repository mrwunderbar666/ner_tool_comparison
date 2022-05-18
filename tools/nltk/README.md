How to: https://www.nltk.org/book/ch07.html

NLTK provides a classifier that has already been trained to recognize named entities, accessed with the function nltk.ne_chunk(). If we set the parameter binary=True [1], then named entities are just tagged as NE; otherwise, the classifier adds category labels such as PERSON, ORGANIZATION, and GPE.

