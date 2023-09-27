# ICEWS Dictionary

- Citation: Boschee, E., Lautenschlager, J., Shellman, S., & Shilliday, A. (2015). icews.actors.20181119.tab. In ICEWS Dictionaries. Harvard Dataverse. https://doi.org/10.7910/DVN/28118/HYSJN6
- Repository: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/28118

## NER Definitions

The official documentation states ("ICEWS Dictionaries Read Me.pdf"):

> "The ICEWS dictionaries contain both named individuals or groups, known as actors, and generic individuals or groups, known as agents. Actors are known by a specific name, such as 'Free Syrian Army' or 'Goodluck Johnathan', while agents are known by a generic improper noun, such as 'insurgents' or 'students'"

# Instructions


Install with

```bash
Rscript tools/icews/get_icews.r
```

Then you can run the evaluation with

```bash
Rscript tools/icews/evaluate.r
```

And test the challenges with

```bash
Rscript tools/icews/challenges.r
```
