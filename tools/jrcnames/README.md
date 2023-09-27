# JRC Names

- Website: https://joint-research-centre.ec.europa.eu/language-technology-resources/jrc-names_en
- Citation: Steinberger, R., Pouliquen, B., Kabadjov, M., Belyaeva, J. & Van Der Goot, E. (2011). JRC-NAMES: A freely available, highly multilingual named entity resource. International Conference Recent Advances in Natural Language Processing, RANLP, 104–110.

## NER Definitions

Very vague:

> "multilingual named entity resource for person and organisation names"

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
