# JRC Names

- Webiste: https://joint-research-centre.ec.europa.eu/language-technology-resources/jrc-names_en
- Citation: Steinberger, R., Pouliquen, B., Kabadjov, M., Belyaeva, J. & Van Der Goot, E. (2011). JRC-NAMES: A freely available, highly multilingual named entity resource. International Conference Recent Advances in Natural Language Processing, RANLP, 104–110.


# Instructions

Install with

```bash
Rscript tools/jrcnames/get_jrc.r
```

Then you can run the evaluation with

```bash
Rscript tools/jrcnames/evaluate.r
```

And test the challenges with

```bash
Rscript tools/jrcnames/challenges.r
```
