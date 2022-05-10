https://huggingface.co/datasets/conll2012_ontonotesv5
https://catalog.ldc.upenn.edu/LDC2013T19


# Obtaining the OntoNotes Release 5.0


While the corpus is free of charge, you are required to create an account and sign a licence agreement.

## Official Procedure

- Register at the [Linguistic Data Consortium](https://catalog.ldc.upenn.edu/signup)
- Make sure you have an academic institution attached to your account. In case your institution is not listed, you have to email the staff to activate your account.
- Go to the Catalog entry for [OntoNotes (LDC2013T19)](https://catalog.ldc.upenn.edu/LDC2013T19) and add it to your cart.
- Sign the agreement, download the file (My Account > Downloads)
- drop the file in this folder: `corpora/ontonotes/ontonotes-release-5.0_LDC2013T19.tgz`
- Run the script `get_ontontes.py`

## Other Means

You can also retrieve the dataset from Huggingface with:

```
git clone https://huggingface.co/datasets/conll2012_ontonotesv5
```