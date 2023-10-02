# KIND (Kessler Italian Named-entities Dataset)

- Repository: https://github.com/dhfbk/KIND
- Website: https://nermud.fbk.eu/
- Citations: 
    - 2023 Edition: NERMuD at EVALITA 2023: Overview of the Named-Entities Recognition on Multi-Domain Documents Task. Alessio Palmero Aprosio and Teresa Paccosi, Proceedings of the Eighth Evaluation Campaign of Natural Language Processing and Speech Tools for Italian. Final Workshop (EVALITA 2023)
    - Original Edition: KIND: an Italian Multi-Domain Dataset for Named Entity Recognition. Teresa Paccosi and Alessio Palmero Aprosio, Proceedings of the 13th Conference on Language Resources and Evaluation 2022 (LREC 2022) [arXiv: 2112.15099](https://arxiv.org/abs/2112.15099)
- Respository: https://github.com/impresso/CLEF-HIPE-2020
- Redistribution granted by licence: The NER annotations are released under the Attribution-NonCommercial 4.0 International (CC BY-NC 4.0) license. Annotation from Alcide De Gasperi's writings are released under the Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0) license



## Source Materials

Mix of freely available news (wikinews) and litarary work.

# Parse the Data

Run

```bash
python corpora/kind/parse_kind.py
```

Does the following:

- unpacks the zip file
- parses the individual tsv datasets
- saves to feather

# Citation

```bibtex
@inproceedings{evalita2023nermud,
    title={{NERMuD} at {EVALITA} 2023: Overview of the Named-Entities Recognition on Multi-Domain Documents Task},
    author={Palmero Aprosio, Alessio and Paccosi, Teresa},
    booktitle={Proceedings of the Eighth Evaluation Campaign of Natural Language Processing and Speech Tools for Italian. Final Workshop (EVALITA 2023)},
    publisher = {CEUR.org},
    year = {2023},
    month = {September},
    address = {Parma, Italy}
}
```