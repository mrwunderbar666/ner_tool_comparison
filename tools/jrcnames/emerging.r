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
  f1 = NULL,
  validation_duration = NULL
)

source('tools/jrcnames/utils.r')

# Load JRC Names Dictionary
jrc_dict <- arrow::read_feather('tools/jrcnames/jrcnames.feather')

# Pre-process: replace token separator '+' with ' '
jrc_dict$keyword <- stringi::stri_replace_all(jrc_dict$keyword, fixed = '+', ' ')

# Load corpus: Emerging Entities (Test)
emerging <- arrow::read_feather('corpora/emerging/emerging.test.annotated.feather')

emerging <- recode_iob(emerging, colname = 'CoNLL_IOB2')

# Transform raw corpus to tcorpus obect
tc_emerging <- corpustools::tokens_to_tcorpus(emerging, doc_col = 'sentence_id', token_id_col = 'token_id', token_col = 'token')

# Run Dictionary over Corpus
start_time <- Sys.time()
tc_emerging$code_dictionary(jrc_dict, case_sensitive = T, token_col = 'token', string_col = 'keyword', sep = ' ', use_wildcards = F)
end_time <- Sys.time()

# Recode Dictionary Results
codings_emerging <- recode_results(tc_emerging$tokens)

# Calculate Evaluation Metrics
cm_emerging <- caret::confusionMatrix(codings_emerging$JRC_NE, reference=codings_emerging$NE, mode = "everything")

r <- cm2df(cm_emerging, 'Emerging-Entities-Test', 'en')
r$evaluation_time <-
  as.double(difftime(end_time, start_time, units = "secs"))

results <- rbind(results, r)

# Save results

if (!dir.exists('results')) {
  dir.create('results')
}

write.csv(results, file='results/jrc_emerging.csv', row.names = F)
