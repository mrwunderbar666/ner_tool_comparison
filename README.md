# Data

- CoNLL 2002
- CoNLL 2003
- Europeana
- OntoNotes
- WikiANN
- CNEC 2.0

## Data Converstion Scripts

Collection of scripts that automatically retrieve the datasets (if possible) and then convert them to a common format. Each corpus is in tokenized long format (one row = one token) and contains the following columns:

- `dataset`: name of dataset
- `language`: language of dataset / tokens
- `subset`: Original name of subset (or split) of dataset. E.g., training, validation, etc.
- `sentence_id`: id of sentence (string), typically enumerated from `000001`. In some cases the corpus also has document ids, then the `sentence_id` includes the `doc_id` as well. E.g, `0001_000001`.
- `token_id`: id (actually position) of token within the sentence. Always starts at 1.
- `token`: actual token in its original form.
- `CoNLL_IOB2`: Named entity tag according to *Inside-Outside-Beginning* scheme as defined by CoNLL. Named entities are limited to Persons, Organizations, Location, and Misc. 

# Evaluation

https://noisy-text.github.io/2017/files/wnuteval.py


# Conclusion

- older tools tend to perform worse on newer copora: but there is a lack of "fresh" data for many languages

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

- [Quaero Broadcast News Extended Named Entity corpus Corpus](http://catalog.elra.info/en-us/repository/browse/ELRA-S0349/)
- [Quaero Old Press Extended Named Entity corpus Corpus](http://catalog.elra.info/en-us/repository/browse/ELRA-W0073/)

## Spanish

- [AnCora](http://clic.ub.edu/corpus/en/ancora) (Catalan & Spanish)

# Difficult Examples

## Nouvel 2016

Page 84:

- He saw John F. Kennedy just before the assassination; 
- His trip through John F. Kennedy went well; 
- The Abrams tank was widely tested during the Gulfwar; 
- The Star Wars VII movie was directed by Abrams; 

## German

- "Und nun will die Bundesregierung Gerhard Schröder Büro und Personal streichen." (tagesthemen, 18.05.20222); boundary challenge 
- "In Corona-Hochburgen wie München und Hamburg nähern sich einige Krankenhäuser der Kapazitätsgrenze." (Spiegel.de, 27.03.2020); neogolism / compund challenge


## English

- This mistaken approach is made plain as Secretary for Commerce and Economic Development Gregory So Kam-leung was recently quoted as saying, "(The government) will try to strike a balance between economic development and the livelihoods of people." (China Daily, HK Edition 03/11/2014 page1); foreign name challenge

- With options running out, the 18-year-old Wong announced at a rally late Monday that he and two other members of his group would go on an indefinite hunger strike to press demands that the Hong Kong government drop restrictions on inaugural 2017 elections for the city’s top leader. (China Post, 2014-12-02); boundary challenge

- Chow Tai Fook plans to double its number of points of sale, including concessionaires and stand-alone outlets, in the next 10 years. (China Daily, 2014-11-24); ambiguity & foreign name challenge
- Founded in 1929 in the southern city of Guangzhou, the jeweler was named after founder Chow Chi-yuen. "Tai Fook" means "big blessing" in Chinese. (China Daily, 2014-11-24); ambiguity & foreign name challenge

- Hong Kong's students tell Xi they don't want a revolution. (Japan Times, 2014-10-12); foreign name challenge
- Agnes Chow Ting, 17, stepped down as a spokeswoman for Scholarism after 2½ years, citing exhaustion, saying that she’s “unable to shoulder such a great burden,” in a post Saturday on Scholarism’s Facebook page. (Japan Times, 2014-10-12); foreign name challenge
- In another sign of tensions in the city, the newspaper Apple Daily, which is owned by prodemocracy activist Jimmy Lai, said on its website late Saturday that a truck was parked outside its headquarters in Tseung Kwan O, blocking access and threatening to prevent the newspaper being distributed. (Japan Times, 2014-10-12); foreign name challenge
