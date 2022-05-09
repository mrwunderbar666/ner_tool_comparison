library(corpustools)
library(arrow)
library(stringi)
library(caret)

# Initialize final results table
results <- data.frame(
  corpus = NULL,
  lang = NULL,
  task = NULL,
  precision = NULL,
  recall = NULL,
  specificity = NULL,
  f1 = NULL
)

### Helper Functions

# Recode CoNLL Corpus Named Entity Columns
recode_iob <- function(df) {
  
  # Recode Code Columns
  # Column for any named entity
  df$any_NE <- as.numeric(df$IOB2 != 'O') 
  
  # Columns without I-* or B-*
  df$NE <- gsub("^[BI]-", "", df$IOB2)
  df$NE <- as.factor(df$NE)
  return(df)
}

# Recode Dictionary Results
recode_results <- function(df) {
  
  codings <- as.data.frame(df)
  
  codings$JRC_any_NE <- as.numeric(!is.na(codings$type))
  codings[is.na(codings$type), ]$type <- ""
  codings$JRC_NE <- "O"
  
  filt_persons <- codings$type == 'P'
  codings[filt_persons, 'JRC_NE'] <- "PER"
  
  filt_orgs <- codings$type == 'O'
  codings[filt_orgs, 'JRC_NE'] <- "ORG"
  
  codings$JRC_NE <- factor(codings$JRC_NE, levels=levels(codings$NE))

  return(codings)  
}

# Extract Relevant Information from caret::confusionMatrix
cm2df <- function(confusion, corpus, lang) {
  return(data.frame(
    corpus = corpus, 
    lang = lang, 
    task = gsub("Class: ", "", names(confusion$byClass[, 'Precision'])), 
    precision = unname(confusion$byClass[, 'Precision']),
    recall = unname(confusion$byClass[, 'Recall']),
    specificity = unname(confusion$byClass[, 'Specificity']),
    f1 = unname(confusion$byClass[, 'F1'])
  ))
}



# Load JRC Names Dictionary
jrc_dict <- arrow::read_feather('tools/jrcnames/jrcnames.feather')

# Pre-process: replace token separator '+' with ' '
jrc_dict$keyword <- stringi::stri_replace_all(jrc_dict$keyword, fixed = '+', ' ')

# Load corpus: CoNLL-2002 Spanish (Test B)
esp <- arrow::read_feather("corpora/conll/esp.testb.feather")

# Load corpus: CoNLL-2002 Dutch (Test B)
ned <- arrow::read_feather("corpora/conll/ned.feather")
ned <- ned[ned$corpus == "ned.testb", ]
ned$sentence <- paste(ned$doc, ned$sentence, sep="_")

# Load corpus: CoNLL-2003 English (Test)
eng <- arrow::read_feather("corpora/conll/conll2003_en_test_iob.feather")

esp <- recode_iob(esp)
ned <- recode_iob(ned)
eng <- recode_iob(eng)

# Transform raw corpus to tcorpus obect
tc_esp <- corpustools::tokens_to_tcorpus(esp, doc_col = 'sentence', token_id_col = 'token_id', token_col = 'token')
tc_ned <- corpustools::tokens_to_tcorpus(ned, doc_col = 'sentence', token_id_col = 'token_id', token_col = 'token')
tc_eng <- corpustools::tokens_to_tcorpus(eng, doc_col = 'sentence_id', token_id_col = 'token_id', token_col = 'token')

# Run Dictionary over Corpus
tc_esp$code_dictionary(jrc_dict, case_sensitive = T, token_col = 'token', string_col = 'keyword', sep = ' ', use_wildcards = F)
tc_ned$code_dictionary(jrc_dict, case_sensitive = T, token_col = 'token', string_col = 'keyword', sep = ' ', use_wildcards = F)
tc_eng$code_dictionary(jrc_dict, case_sensitive = T, token_col = 'token', string_col = 'keyword', sep = ' ', use_wildcards = F)

# Recode Dictionary Results
codings_esp <- recode_results(tc_esp$tokens)
codings_ned <- recode_results(tc_ned$tokens)
codings_eng <- recode_results(tc_eng$tokens)

# Calculate Evaluation Metrics
cm_esp <- caret::confusionMatrix(codings_esp$JRC_NE, reference=codings_esp$NE, mode = "everything")

r <- cm2df(cm_esp, 'CoNLL-2002-esp-testb', 'es')

results <- rbind(results, r)

cm_esp <- caret::confusionMatrix(as.factor(codings_esp$JRC_any_NE), reference=as.factor(codings_esp$any_NE), mode = "everything")

results <- rbind(results, data.frame(
  corpus = 'CoNLL-2002-esp-testb', 
  lang = 'es', 
  task = 'any_NE', 
  precision = cm_esp$byClass['Precision'],
  recall = cm_esp$byClass['Recall'],
  specificity = cm_esp$byClass['Specificity'],
  f1 = cm_esp$byClass['F1']
))


cm_ned <- caret::confusionMatrix(codings_ned$JRC_NE, reference=codings_ned$NE, mode = "everything")
r <- cm2df(cm_ned, 'CoNLL-2002-ned-testb', 'nl')
results <- rbind(results, r)

cm_ned <- caret::confusionMatrix(as.factor(codings_ned$JRC_any_NE), reference=as.factor(codings_ned$any_NE), mode = "everything")


results <- rbind(results, data.frame(
  corpus = 'CoNLL-2002-ned-testb', 
  lang = 'nl', 
  task = 'any_NE', 
  precision = cm_esp$byClass['Precision'],
  recall = cm_esp$byClass['Recall'],
  specificity = cm_esp$byClass['Specificity'],
  f1 = cm_esp$byClass['F1']
))


cm_eng <- caret::confusionMatrix(codings_eng$JRC_NE, reference=codings_eng$NE, mode = "everything")

r <- cm2df(cm_eng, 'CoNLL-2003-eng-test', 'en')

results <- rbind(results, r)

cm_eng <- caret::confusionMatrix(as.factor(codings_eng$JRC_any_NE), reference=as.factor(codings_eng$any_NE), mode = "everything")

results <- rbind(results, data.frame(
  corpus = 'CoNLL-2003-eng-test', 
  lang = 'en', 
  task = 'any_NE', 
  precision = cm_esp$byClass['Precision'],
  recall = cm_esp$byClass['Recall'],
  specificity = cm_esp$byClass['Specificity'],
  f1 = cm_esp$byClass['F1']
))

# Save results

write.csv(results, file='results/jrc_conll2002.csv', row.names = F)
