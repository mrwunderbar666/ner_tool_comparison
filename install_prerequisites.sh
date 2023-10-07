# Get Script for CoNLL 2003 English Dataset from huggingface
curl https://huggingface.co/datasets/conll2003/raw/main/conll2003.py -o utils/conll2003.py

# Install Python Packages
python3 -m pip install -r requirements.txt

# Install R Packages
Rscript r_packages.r

# Install Tools
python3 tools/corenlp/get_corenlp.py
Rscript tools/icews/get_icews.r
python3 tools/jrcnames/get_jrc.py
python3 tools/nltk/get_dependencies.py
python3 tools/opennlp/get_opennlp.py

python3 -m spacy download zh_core_web_lg-3.3.0 --direct
python3 -m spacy download zh_core_web_trf-3.3.0 --direct
python3 -m spacy download nl_core_news_lg-3.3.0 --direct
python3 -m spacy download en_core_web_lg-3.3.0 --direct
python3 -m spacy download en_core_web_trf-3.3.0 --direct
python3 -m spacy download fr_core_news_lg-3.3.0 --direct
python3 -m spacy download de_core_news_lg-3.3.0 --direct
python3 -m spacy download es_core_news_lg-3.3.0 --direct
python3 -m spacy download xx_ent_wiki_sm-3.3.0 --direct
python3 -m spacy download pt_core_news_lg-3.3.0 --direct
python3 -m spacy download fi_core_news_lg-3.3.0 --direct
python3 -m spacy download ca_core_news_lg-3.3.0 --direct
python3 -m spacy download it_core_news_lg-3.3.0 --direct
