# LÂMPADA - Second HAREM Resource Package

- Citations: 
    - LÂMPADA - Second HAREM Resource Package (Preferred Corpus Name)
    - Cristina Mota & Diana Santos (eds.). Desafios na avaliação conjunta do reconhecimento de entidades mencionadas: O Segundo HAREM. Linguateca, 2008. https://www.linguateca.pt/LivroSegundoHAREM/ (Accompanying publication that includes documentation)
- Website: https://www.linguateca.pt/HAREM/
- Licence: Redistributed as granted by the Creative Commons Licence 3.0, see https://creativecommons.org/licenses/by/3.0/ie/

## Source Materials

The corpus covers two varieties of Portuguese: Brazil and Portugal. 
It contains texts from news, education, opinion pieces, blogs, FAQ section, legal texts, literary texts, and advertisements.

# Parse the Data

Run

```bash
python corpora/ancora/parse_ancora.py
```

Does the following:

- unpacks the zip file
- parses the individual XML documents
- generates a training, test, and validation split


# Acknowledgments

As stated in the original readme file:

> Linguateca is jointly funded by the Portuguese Government and the European Union (FEDER and FSE) under contract ref. POSC/339/1.3/C/NAC, and by FCCN and by UMIC. 
> We are grateful to Jorge Baptista, Caroline Hagège and Nuno Mamede for introducing the TEMPO track, to Nuno Cardoso for deploying SAHARA, and to Luís Miguel Cabral, Luís Costa and David Cruz for their support in several other tasks.
> We also thank all Second HAREM participants for granting us permission to further distribute their runs. 

