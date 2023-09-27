# Stanford CoreNLP

- Citation CoreNLP Pipeline: Manning, Christopher D., Mihai Surdeanu, John Bauer, Jenny Finkel, Steven J. Bethard, and David McClosky. 2014. The Stanford CoreNLP Natural Language Processing Toolkit In Proceedings of the 52nd Annual Meeting of the Association for Computational Linguistics: System Demonstrations, pp. 55-60.
- Citation NER Component: Jenny Rose Finkel, Trond Grenager, and Christopher Manning. 2005. Incorporating Non-local Information into Information Extraction Systems by Gibbs Sampling. Proceedings of the 43nd Annual Meeting of the Association for Computational Linguistics (ACL 2005), pp. 363-370. http://nlp.stanford.edu/~manning/papers/gibbscrf3.pdf
- Official Websites: 
    - https://stanfordnlp.github.io/CoreNLP/
    - https://nlp.stanford.edu/software/CRF-NER.html
- Algorithm: Conditional Random Fields enhanced with Gibbs Sampling

## Training Material

- English: "Our big English NER models were trained on a mixture of CoNLL, MUC-6, MUC-7 and ACE named entity corpora, and as a result the models are fairly robust across domains."
    - MUC-6 (copyrighted corpus): Chinchor, Nancy, and Beth Sundheim. Message Understanding Conference (MUC) 6 LDC2003T13. Web Download. Philadelphia: Linguistic Data Consortium, 2003. https://doi.org/10.35111/wbcc-y063
        - Annotation guidlines in the file "ne-task-def.v2.1.ps"
        - "This subtask [named entities] is limited to proper names, acronyms, and perhaps miscellaneous other unique identifiers, which are categorized via the TYPE attribute as follows:"
            - "ORGANIZATION: named corporate, governmental, or other organizational entity"
            - "PERSON: named person or family"
            - "LOCATION: name of politically or geographically defined location (cities, provinces, countries, international regions, bodies of water, mountains, etc.)
    - MUC-7 (copyrighted corpus): Chinchor, Nancy. Message Understanding Conference (MUC) 7 LDC2001T02. Web Download. Philadelphia: Linguistic Data Consortium, 2001. https://doi.org/10.35111/jygm-3h55
        - Annotation guidlines in the file "guidelines.NEtaskdef.3.5.ps"
        - Same annotation rules as for MUC-6
    - ACE (copyrighted corpus): Mitchell, Alexis, et al. ACE-2 Version 1.0 LDC2003T11. Web Download. Philadelphia: Linguistic Data Consortium, 2003. (https://doi.org/10.35111/kcqk-v224)
        - Annotataion guidelines in the file "EDT-Guidelines-V2-5-1.PDF"
        - "An entity is an object or set of objects in the world.  A mention is a reference to an entity.  Entities may be referenced by their name, indicated by a common noun or noun phrase, or represented by a pronoun."
            - "Person - Person entities are limited to humans.  A person may be a single individual or a group."
            - "Organization - Organization entities are limited to corporations, agencies, and other groups of people defined by an established organizational structure."
            - "Facility - Facility entities are limited to buildings and other permanent man-made structures and real estate improvements."
            - "Location - Location entities are limited to geographical entities such as geographical areas and landmasses, bodies of water, and geological formations."
            - "GPE (Geo-political Entity) - GPE entities are geographical regions defined by political and/or social groups.  A GPE entity subsumes and does not distinguish between a nation, its region, its government, or its people."


- Chinese: "Ontonotes Chinese named entity data"
- French: Not documented
- German: CoNLL 2003 data
- Spanish: Not documented
- Italian: Not documented
- Hungarian: Not documented

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
