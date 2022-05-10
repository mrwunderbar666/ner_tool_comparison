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

source('tools/jrcnames/utils.r')

# Load JRC Names Dictionary
jrc_dict <- arrow::read_feather('tools/jrcnames/jrcnames.feather')

# Pre-process: replace token separator '+' with ' '
jrc_dict$keyword <- stringi::stri_replace_all(jrc_dict$keyword, fixed = '+', ' ')

# Load corpus: Emerging Entities (Test)
germeval <- arrow::read_feather('corpora/germeval2014/NER-de-test.feather')
germeval$token_id <- as.numeric(germeval$token_id)

germeval <- recode_iob(germeval, colname = 'CoNLL_IOB2')

# Transform raw corpus to tcorpus obect
tc_germeval <- corpustools::tokens_to_tcorpus(germeval, doc_col = 'sentence', token_id_col = 'token_id', token_col = 'token')

# Run Dictionary over Corpus
tc_germeval$code_dictionary(jrc_dict, case_sensitive = T, token_col = 'token', string_col = 'keyword', sep = ' ', use_wildcards = F)

# Recode Dictionary Results
codings_germeval <- recode_results(tc_germeval$tokens)

# Calculate Evaluation Metrics
cm_germeval <- caret::confusionMatrix(codings_germeval$JRC_NE, reference=codings_germeval$NE, mode = "everything")

r <- cm2df(cm_germeval, 'germeval2014-test', 'de')

results <- rbind(results, r)

cm_germeval <- caret::confusionMatrix(as.factor(codings_germeval$JRC_any_NE), reference=as.factor(codings_germeval$any_NE), mode = "everything")

results <- rbind(results, data.frame(
  corpus = 'germeval2014-test', 
  lang = 'de', 
  task = 'any_NE', 
  precision = cm_germeval$byClass['Precision'],
  recall = cm_germeval$byClass['Recall'],
  specificity = cm_germeval$byClass['Specificity'],
  f1 = cm_germeval$byClass['F1']
))


# Save results

if (!dir.exists('results')) {
  dir.create('results')
}

write.csv(results, file='results/jrc_germeval.csv', row.names = F)
