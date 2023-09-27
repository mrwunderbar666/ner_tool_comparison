# spaCy

- Website: https://spacy.io/
- Algorithm: Neural Networks

## Training Material

- English: OntoNotes 5
- Chinese: OntoNotes 5
- Dutch: Dutch NER Annotations for UD LassySmall (NLP Town) [proprietary]
- French: WikiNER
- German: WikiNER
- Spanish: WikiNER
- Multilingual: WikiNER

# Instructions


Get models with

```bash
./tools/spacy/download_models.sh
```

Then you can run the evaluation with

```bash
python tools/spacy/evaluate.py
```

And test the challenges with

```bash
python tools/spacy/challenges.py
```


# Models

## Chinese

- `zh_core_web_lg`
- `zh_core_web_trf`

## Dutch

- `nl_core_news_lg`

## English

- `en_core_web_lg`: English pipeline optimized for CPU. Components: tok2vec, tagger, parser, senter, ner, attribute_ruler, lemmatizer.
- `en_core_web_trf`: English transformer pipeline (roberta-base). Components: transformer, tagger, parser, ner, attribute_ruler, lemmatizer.

## French

- `fr_core_news_lg`
- `fr_dep_news_trf`: no support for NER

## German

- `de_core_news_lg`
- `de_dep_news_trf`

## Spanish

- `es_core_news_lg`
- `es_dep_news_trf`: no support for NER

## Multilingual

- `xx_ent_wiki_sm`
