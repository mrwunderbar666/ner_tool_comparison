# CLEF-HIPE-2020 - Shared Task Data (Version 1.4)

HIPE (Identifying Historical People, Places and other Entities) is a named entity processing evaluation campaign on historical newspapers in French, German and English, organized in the context of the impresso project and run as a CLEF 2020 Evaluation Lab.


- Website: https://impresso.github.io/CLEF-HIPE-2020/
- Citation: 
    - M. Ehrmann, M. Romanello, A. Flückiger, and S. Clematide, Extended Overview of CLEF HIPE 2020: Named Entity Processing on Historical Newspapers in Working Notes of CLEF 2020 - Conference and Labs of the Evaluation Forum, Thessaloniki, Greece, 2020, vol. 2696, p. 38. [doi: 10.5281/zenodo.4117566](https://doi.org/10.5281/zenodo.4117566).
    - Link to publication: https://infoscience.epfl.ch/record/281054
- Redistribution granted by licence: The HIPE datasets are licensed under a Creative Commons Attribution-NonCommercial 4.0 International License.
- Respository: https://github.com/impresso/CLEF-HIPE-2020



## Source Materials

Historical newspapers in German, French and English, which cover about 150 years. Source material
as digitized with OCR and some noise was manually corrected.

# Parse the Data

Run

```bash
python corpora/hipe/parse_hipe.py
```

Does the following:

- unpacks the zip file
- parses the individual tsv datasets
- applies mappings
- saves to feather


# Citation

```bibtex
@inproceedings{ehrmann_extended_2020,
  title = {Extended {Overview} of {CLEF HIPE} 2020: {Named Entity Processing} on {Historical Newspapers}},
  booktitle = {{CLEF 2020 Working Notes}. {Working Notes} of {CLEF} 2020 - {Conference} and {Labs} of the {Evaluation Forum}},
  author = {Ehrmann, Maud and Romanello, Matteo and Fl{\"u}ckiger, Alex and Clematide, Simon},
  editor = {Cappellato, Linda and Eickhoff, Carsten and Ferro, Nicola and N{\'e}v{\'e}ol, Aur{\'e}lie},
  year = {2020},
  volume = {2696},
  pages = {38},
  publisher = {{CEUR-WS}},
  address = {{Thessaloniki, Greece}},
  doi = {10.5281/zenodo.4117566},
  url = {https://infoscience.epfl.ch/record/281054},
}
```