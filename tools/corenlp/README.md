# Stanford CoreNLP

- Citation CoreNLP Pipeline: Manning, Christopher D., Mihai Surdeanu, John Bauer, Jenny Finkel, Steven J. Bethard, and David McClosky. 2014. The Stanford CoreNLP Natural Language Processing Toolkit In Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics: System Demonstrations, pp. 55-60.
- Citation NER Component: Jenny Rose Finkel, Trond Grenager, and Christopher Manning. 2005. Incorporating Non-local Information into Information Extraction Systems by Gibbs Sampling. Proceedings of the 43nd Annual Meeting of the Association for Computational Linguistics (ACL 2005), pp. 363-370. http://nlp.stanford.edu/~manning/papers/gibbscrf3.pdf
- Official Websites: 
    - https://stanfordnlp.github.io/CoreNLP/
    - https://nlp.stanford.edu/software/CRF-NER.html
- Algorithm: Conditional Random Fields enhanced with Gibbs Sampling

## Training Material

- English: "Our big English NER models were trained on a mixture of CoNLL, MUC-6, MUC-7 and ACE named entity corpora, and as a result the models are fairly robust across domains."
- Chinese: "Ontonotes Chinese named entity data"
- French: Not documented
- German: CoNLL 2003 data
- Spanish: Not documented
- Italian: Not documented
- Hungarian: Not documented
- Arabic: Not documented

# Instructions

Install with

```bash
python tools/corenlp/get_corenlp.py
```

Then you can run the evaluation with

```bash
python tools/corenlp/evaluate.py
```

And test the challenges with

```bash
python tools/corenlp/challenges.py
```
