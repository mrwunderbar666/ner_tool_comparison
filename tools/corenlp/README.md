# Stanford CoreNLP

- Citation CoreNLP Pipeline: Manning, Christopher D., Mihai Surdeanu, John Bauer, Jenny Finkel, Steven J. Bethard, and David McClosky. 2014. The Stanford CoreNLP Natural Language Processing Toolkit In Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics: System Demonstrations, pp. 55-60.
- Citation NER Component: Jenny Rose Finkel, Trond Grenager, and Christopher Manning. 2005. Incorporating Non-local Information into Information Extraction Systems by Gibbs Sampling. Proceedings of the 43nd Annual Meeting of the Association for Computational Linguistics (ACL 2005), pp. 363-370. http://nlp.stanford.edu/~manning/papers/gibbscrf3.pdf
- Official Website: https://stanfordnlp.github.io/CoreNLP/
- Algorithm: Conditional Random Fields enhanced with Gibbs Sampling

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
