# Collection of all mappings
# different corpora / tools use different labels for NER

###############################
# Corpora
###############################

# CNEC 2.0
# https://ufal.mff.cuni.cz/~strakova/cnec2.0/ne-type-hierarchy.pdf
# Very fine grained labelling (see their documentation)
# Requires hierarchical conversion (from outer most to inner most unit)
# NE Containers
# P: complex person names -> PER
# T: complex time expressions -> O
# A: complex address expressions -> O/LOC
# C: complex bibliographic expressions -> O/MISC

# Types of NE
# a: Numbers in addresses -> O
# g: Geographical names -> LOC
# i: Institutions -> ORG
# m: Media names -> MISC
# n: Number expressions -> O
# o: Artifact names -> MISC
# p: Personal names -> PER
# t: Time expressions -> O

cnec2conll = {'P': 'PER', 'p.*': 'PER', 
            'i.*': 'ORG', 
            'g.*': 'LOC', 
            # everything that is not Person, institution, time, numbers, or addresses
            '^[^OPigTtna].*': 'MISC',
            # drop numerical expressions
            'T': 'O', 'n.*': 'O', 't.*': 'O', 'a.*': 'O'} 


# Emerging Entities
# Documentation: http://noisy-text.github.io/2017/files/README.md
# Person -> PER
# Location (including GPE, facility) -> LOC
# Corporation -> ORG
# Consumer good (tangible goods, or well-defined services) -> MISC
# Creative work (song, movie, book, and so on) -> MISC
# Group (subsuming music band, sports team, and non-corporate organisations) -> ORG

emerging2conll = {'person': 'PER', 
                  'group': 'ORG', 
                  'corporation': 'ORG',
                  'location': 'LOC', 
                  'creative-work': 'MISC', 
                  'product': 'MISC'
                  }

# GermEval 2014
# Almost identical to CoNLL
# OTH -> MISC
# Unnesting: 
# e.g. "Eurospeedway Lausitz"
#       <B-ORG>      <I-ORG <B-LOC>> -> <B-ORG> <I-ORG>


# Ontonotes 5.0
# See file: ontonotes-named-entity-guidelines-v14.pdf

# Person Name (PERSON) -> PER
# Nationality, Other, Religion, Political (NORP) -> MISC
# Facility (FAC) -> LOC
# Organization (ORG) -> ORG
# Geographical/Social/Political Entity (GPE) -> LOC
# Location (LOC) -> LOC
# Product (PRODUCT) -> MISC
# Date (DATE) -> O
# Time (TIME) -> O
# Percent (PERCENT) -> O
# Money (MONEY) -> O
# Quantity (QUANTITY) -> O 
# Ordinal (ORDINAL) -> O
# Cardinal (CARDINAL) -> O
# Event (EVENT) -> MISC
# Work of Art (WORK_OF_ART) -> MISC
# Law (LAW) -> MISC
# Language (LANGUAGE) -> MISC

ontonotes2conll = {'I-PERSON': 'I-PER', 'B-PERSON': 'B-PER',
                    'I-GPE': 'I-LOC', 'B-GPE': 'B-LOC',
                    'I-FAC': 'I-LOC', 'B-FAC': 'B-LOC',
                    'I-EVENT': 'I-MISC', 'B-EVENT': 'B-MISC', 
                    'I-WORK_OF_ART': 'I-MISC', 'B-WORK_OF_ART': 'B-MISC', 
                    'I-PRODUCT': 'I-MISC', 'B-PRODUCT': 'B-MISC', 
                    'I-LAW': 'I-MISC', 'B-LAW': 'B-MISC', 
                    'I-NORP': 'I-MISC', 'B-NORP': 'B-MISC', 
                    'I-LANGUAGE': 'I-MISC', 'B-LANGUAGE': 'B-MISC', 
                    '[BI]-DATE': 'O', 
                    '[BI]-TIME': 'O', 
                    '[BI]-CARDINAL': 'O', 
                    '[BI]-MONEY': 'O', 
                    '[BI]-PERCENT': 'O', 
                    '[BI]-ORDINAL': 'O', 
                    '[BI]-QUANTITY': 'O'}

# WikiANN
# Same as CoNLL

# AnCora
# Mapping for strong named entities
# See AnCora documentation, table 11 (p. 37)
ancora2conll = {'person': 'PER',
                     'organization': 'ORG',
                     'location': 'LOC',
                     'other': 'MISC'
                     }

# Harem
# https://www.linguateca.pt/aval_conjunta/HAREM/GlossarioReRelEMIngles.html

harem2conll = {"PESSOA": "PER",
                "TEMPO": "O", # Time
                "LOCAL": "LOC", # Place
                "ORGANIZACAO": "ORG",
                "OBRA": "O", # TITLE
                "ACONTECIMENTO": "MISC", # Event
                "VALOR": "O", # Amounts 
                "ABSTRACCAO": "MISC", # Abstract
                "COISA": "MISC", # Thing
                "OUTRO": "MISC" # Other
                }

# FiNER
# see paper: 
# https://doi.org/10.1007/s10579-019-09471-7
# https://arxiv.org/abs/1908.04212

finer2conll = {'B-DATE': 'O',
                'I-DATE': 'O',
                'B-PRO': 'B-MISC',
                'I-PRO': 'I-MISC',
                'B-EVENT': 'B-MISC',
                'I-EVENT': 'I-MISC'}


# HIPE 
# CLEF HIPE 2020: Named Entity Processing on Historical Newspapers
# see paper:
# https://infoscience.epfl.ch/record/281054

hipe2conll = {'B-pers': 'B-PER',
              'I-pers': 'I-PER',
              'B-org': 'B-ORG',
              'I-org': 'I-ORG',
              'B-loc': 'B-LOC', 
              'I-loc': 'I-LOC',
              'B-prod': 'B-MISC',
              'I-prod': 'I-MISC',
              'B-time': 'O',
              'I-time': 'O'
              }

# SoNaR
# See documentation: 
# https://taalmaterialen.ivdnt.org/wp-content/uploads/documentatie/sonar_documentatie.pdf

# 'O', 
# Persons: 'B-per', 'I-per'
# Organizations: 'B-org', 'I-org'
# Locations: 'B-loc', 'I-loc'
# Misc: 'B-misc', 'I-misc'
# Products: 'B-pro', 'I-pro'
# Events (natural and human): 'B-eve', 'I-eve'

# first uppercase everything
# then turn Products / Events into Misc
# Per, Org, Loc match CoNLL definitions

sonar2conll = {'B-PRO': 'B-MISC', 'I-PRO': 'I-MISC',
               'B-EVE': 'B-MISC', 'I-EVE': 'I-MISC'}


###############################
# Tools
###############################

# CoreNLP
# unclear documentation, educated guesses are required for mapping

# don't change: 'B-MISC', 'I-MISC'
corenlp2conll = {'B-ORGANIZATION': 'B-ORG', 'I-ORGANIZATION': 'I-ORG',
                    'B-PERSON': 'B-PER', 'I-PERSON': 'I-PER', 
                    'B-LOCATION': 'B-LOC', 'I-LOCATION': 'I-LOC',
                    'B-CITY': 'B-LOC', 'I-CITY': 'I-LOC', 
                    'B-COUNTRY': 'B-LOC', 'I-COUNTRY': 'I-LOC', 
                    'B-STATE_OR_PROVINCE': 'B-LOC', 'I-STATE_OR_PROVINCE': 'I-LOC', 
                    'B-GPE': 'B-LOC', 'I-GPE': 'I-LOC',
                    'B-FACILITY': 'B-LOC', 'I-FACILITY': 'I-LOC',
                    'B-RELIGION': 'B-MISC', 'I-RELIGION': 'I-MISC',
                    'B-NATIONALITY': 'B-MISC', 'I-NATIONALITY': 'I-MISC',
                    'B-DEMONYM': 'B-MISC', 'I-DEMONYM': 'I-MISC',
                    'B-CAUSE_OF_DEATH': 'B-MISC', 'I-CAUSE_OF_DEATH': 'I-MISC',
                    'B-IDEOLOGY': 'B-MISC', 'I-IDEOLOGY': 'I-MISC',
                    'B-TITLE': 'O', 'I-TITLE': 'O', 
                    'B-MONEY': 'O', 'I-MONEY': 'O', 
                    'B-PERCENT': 'O', 'I-PERCENT': 'O', 
                    'B-NUMBER': 'O', 'I-NUMBER': 'O', 
                    'B-DURATION': 'O', 'I-DURATION': 'O',
                    'B-TIME': 'O', 'I-TIME': 'O', 
                    'B-ORDINAL': 'O', 'I-ORDINAL': 'O', 
                    'B-DATE': 'O', 'I-DATE': 'O', 
                    'B-CRIMINAL_CHARGE': 'O', 'I-CRIMINAL_CHARGE': 'O', 
                    'B-SET': 'O', 'I-SET': 'O', 
                    'B-EMAIL': 'O', 'I-EMAIL': 'O',
                    'B-URL': 'O', 'I-URL': 'O'
                    }

# Frog (Dutch)
# Uses SoNaR scheme (above)

# Nametagger (Czech model)
# uses annotation scheme of CNEC

nametagger2conll = {
                    'B-P': 'B-PER',
                    'I-P': 'I-PER', 
                    'B-p.*': 'B-PER', 
                    'I-p.*': 'I-PER', 
                    'B-i.*': 'B-ORG', 
                    'I-i.*': 'I-ORG', 
                    'B-g.*': 'B-LOC', 
                    'I-g.*': 'I-LOC', 
                    # everything that is not Person, institution, time, numbers, or addresses
                    'B-[^OPigTtna].*': 'B-MISC',
                    'I-[^OPigTtna].*': 'I-MISC',
                    # drop numerical expressions
                    '[BI]-T': 'O', '[BI]-n.*': 'O', '[BI]-t.*': 'O', '[BI]-a.*': 'O'} 


# NLTK 

nltk2conll = {'B-GPE': 'B-LOC', 'I-GPE': 'I-LOC', 
              'B-GSP': 'B-LOC', 'I-GSP': 'I-LOC', 
              'B-FACILITY': 'B-LOC', 'I-FACILITY': 'I-LOC', 
              'B-LOCATION': 'B-LOC', 'I-LOCATION': 'I-LOC', 
              'B-PERSON': 'B-PER', 'I-PERSON': 'I-PER',
              'B-ORGANIZATION': 'B-ORG', 'I-ORGANIZATION': 'I-ORG', 
              'B-PERSON': 'B-PER', 'I-PERSON': 'I-PER'}

# OpenNLP

opennlp2conll = {'B-location': 'B-LOC', 'I-location': 'I-LOC',
                'B-person': 'B-PER', 'I-person': 'I-PER',
                'B-organization': 'B-ORG', 'I-organization': 'I-ORG',
                'B-misc': 'B-MISC', 'I-misc': 'I-MISC'}

# SpaCy

# ORG not changed
spacy2conll = {
        'B-GPE': 'B-LOC', 'I-GPE': 'I-LOC',
        'B-FAC': 'B-LOC', 'I-FAC': 'I-LOC', 
        'B-PERSON': 'B-PER', 'I-PERSON': 'I-PER',
        'B-NORP': 'B-MISC', 'I-NORP': 'I-MISC',
        'B-WORK_OF_ART': 'B-MISC', 'I-WORK_OF_ART': 'B-MISC',
        'B-LANGUAGE': 'B-MISC', 'I-LANGUAGE': 'I-MISC',
        'B-PRODUCT': 'B-MISC', 'I-PRODUCT': 'I-MISC', 
        'B-EVENT': 'B-MISC', 'I-EVENT': 'I-MISC', 
        'B-LAW': 'B-MISC', 'I-LAW': 'I-MISC',
        'B-CARDINAL': 'O', 'I-CARDINAL': 'O',
        'B-DATE': 'O', 'I-DATE': 'O', 
        'B-TIME': 'O', 'I-TIME': 'O',
        'B-ORDINAL': 'O', 'I-ORDINAL': 'O', 
        'B-QUANTITY': 'O', 'I-QUANTITY': 'O',
        'B-MONEY': 'O', 'I-MONEY': 'O', 
        'B-PERCENT': 'O', 'I-PERCENT': 'O',
}