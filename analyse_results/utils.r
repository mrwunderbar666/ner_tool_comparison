
corpus_release_year <- c("conll" = "2002", 
                        "ontonotes" = "2013", 
                        "germeval2014" = "2014", 
                        "cnec2.0" = "2014", 
                        "europeana" = "2016", 
                        "emerging" =  "2017", 
                        "wikiann" = "2017",
                        "kind" = "2020",
                        "ancora" = "2008",
                        "hipe" = "2020",
                        "sonar" = "2013",
                        "aqmar" = "2012",
                        "finer" = "2020",
                        "nerkor" = "2021",
                        "harem" = "2008")

pretty_corpora <- c(
                    "ancora" = "AnCora",
                    "aqmar" = "AQMAR",
                    "cnec2.0" = "CNEC 2.0", 
                    "conll" = "CoNLL", 
                    "emerging" = "Emerging Entities", 
                    "europeana" = "Europeana", 
                    "finer" = "FiNER",
                    "germeval2014" = "GermEval", 
                    "harem" = "HAREM",
                    "hipe" = "HIPE",
                    "kind" = "KIND",
                    "nerkor" = "NYTK-NerKor",
                    "ontonotes" = "OntoNotes", 
                    "wikiann" = "WikiANN",
                    "sonar" = "SoNaR")


tools_pretty <-   c("xlmroberta" = "XLM-RoBERTa",
                    "spacy_trf" = "spaCy (transformers)",
                    "spacy_lg" = "spaCy",
                    "opennlp" = "OpenNLP",
                    "nltk" = "NLTK",
                    "nametagger" = "Nametagger",
                    "jrc" = "JRC Names",
                    "icews" = "ICEWS", 
                    "frog" = "Frog",
                    "corenlp" = "CoreNLP")

language_codes <- c("ar" = "Arabic",
                    "ca" = "Catalan",
                    "cs" = "Czech",
                    "de" = "German",
                    "en" = "English",
                    "es" = "Spanish",
                    "fi" = "Finnish",
                    "fr" = "French",
                    "hu" = "Hungarian",
                    "it" = "Italian",
                    "nl" = "Dutch",
                    "pt" = "Portuguese",
                    "zh" = "Chinese")


tasks_pretty <- c("PER" = "Persons",
                  "ORG" = "Organizations",
                  "LOC" = "Locations",
                  "MISC" = "Misc",
                  "overall" = "Overall")