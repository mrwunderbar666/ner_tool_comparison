echo "Running challenges ..."

# corenlp
echo "CoreNLP..."
python3 tools/corenlp/challenges.py
echo "------------------------------------------------------------"

# frog
echo "Frog..."
python3 tools/frog/challenges.py
echo "------------------------------------------------------------"

# icews
echo "ICEWS..."
Rscript tools/icews/challenges.r
echo "------------------------------------------------------------"

# jrcnames
echo "JRC Names..."
Rscript tools/jrcnames/challenges.r
echo "------------------------------------------------------------"

# nametagger
echo "nametagger..."
Rscript tools/nametagger/challenges.r
echo "------------------------------------------------------------"

# nltk
echo "NLTK..."
python3 tools/nltk/challenges.py
echo "------------------------------------------------------------"

# opennlp
echo "OpenNLP..."
python3 tools/opennlp/challenges.py
echo "------------------------------------------------------------"

# spacy
echo "spaCy..."
python3 tools/spacy/challenges.py
echo "------------------------------------------------------------"

# xlm-roberta
echo "XLM-RoBERTa..."
python3 tools/xlmroberta/challenges.py
echo "------------------------------------------------------------"

echo "Done!"