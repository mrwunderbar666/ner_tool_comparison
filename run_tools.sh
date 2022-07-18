# corenlp
python3 tools/corenlp/get_corenlp.py
python3 tools/corenlp/evaluate.py

# icews
Rscript tools/icews/get_icews.r
Rscript tools/icews/evaluate.r

# jrcnames
python3 tools/jrcnames/get_jrc.py
Rscript tools/jrcnames/evaluate.r

# nametagger
Rscript tools/nametagger/english.r
Rscript tools/nametagger/czech.r
python3 tools/nametagger/evaluate.py

# nltk
python3 tools/nltk/get_dependencies.py
python3 tools/nltk/english.py

# opennlp
python3 tools/opennlp/get_opennlp.py
python3 tools/opennlp/evaluate.py

# spacy
python3 -m spacy download zh_core_web_lg
python3 -m spacy download zh_core_web_trf
python3 -m spacy download nl_core_news_lg
python3 -m spacy download en_core_web_lg
python3 -m spacy download fr_core_news_lg
python3 -m spacy download de_core_news_lg
python3 -m spacy download es_core_news_lg
python3 -m spacy download xx_ent_wiki_sm

python3 tools/spacy/evaluate.py