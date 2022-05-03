library(corpustools)
library(arrow)
library(stringi)
setwd("~/development/ner_study/tools/jrcnames")

df <- arrow::read_feather('jrcnames.feather')
df$keyword <- stringi::stri_replace_all(df$keyword, fixed = '+', ' ')

esp <- read_feather("/home/balluff/development/ner_study/datasets/conll/esp.testa.feather")

tc <- tokens_to_tcorpus(esp, doc_col = 'sentence', token_id_col = 'token_id', token_col = 'token')

tc$code_dictionary(df, case_sensitive = T, token_col = 'token', string_col = 'keyword', sep = ' ', use_wildcards = F)

