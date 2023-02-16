# OntoNotes Release 5.0

- Citation: Weischedel, Ralph, Palmer, Martha, Marcus, Mitchell, Hovy, Eduard, Pradhan, Sameer, Ramshaw, Lance, Xue, Nianwen, Taylor, Ann, Kaufman, Jeff, Franchini, Michelle, El-Bachouti, Mohammed, Belvin, Robert & Houston, Ann. (2013). OntoNotes Release 5.0. https://doi.org/10.35111/XMHB-2B84
- Official repository: https://catalog.ldc.upenn.edu/LDC2013T19
- Huggingface link: https://huggingface.co/datasets/conll2012_ontonotesv5

# Obtaining the OntoNotes Release 5.0


While the corpus is free of charge, you are required to create an account and sign a licence agreement.

## Official Procedure

- Register at the [Linguistic Data Consortium](https://catalog.ldc.upenn.edu/signup)
- Make sure you have an academic institution attached to your account. In case your institution is not listed, you have to email the staff to activate your account.
- Go to the Catalog entry for [OntoNotes (LDC2013T19)](https://catalog.ldc.upenn.edu/LDC2013T19) and add it to your cart.
- Sign the agreement, download the file (My Account > Downloads)

## Converting the raw data

- Get [this excellent parsing script](https://github.com/nsu-ai/ontonotes-5-parsing) to convert the single files into one JSON corpus
- drop the resulting file in this folder: `corpora/ontonotes/ontonotes5_parsed.json`
- Run the script `convert_ontontes.py`

## Other Means

You can also retrieve the dataset from Huggingface with:

```
git clone https://huggingface.co/datasets/conll2012_ontonotesv5
```