# FiNER - A Finnish news corpus for named entity recognition

- Website: https://github.com/mpsilfve/finer-data
- Citation: 
    - Ruokolainen, T., Kauppinen, P., Silfverberg, M. et al. A Finnish news corpus for named entity recognition. Lang Resources & Evaluation 54, 247–272 (2020). [10.1007/s10579-019-09471-7](https://doi.org/10.1007/s10579-019-09471-7)
    - ArXiv link: https://arxiv.org/abs/1908.04212
- Licence: CC BY-ND-NC 1.0 


# Parse the Data

Run

```bash
python corpora/finer/get_finer.py
```

Does the following:

- downloads text files from official github repository
- parses each file
- applies mapping
