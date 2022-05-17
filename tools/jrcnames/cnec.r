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

# Load corpus: CNEC 2.0 (DTest)
cnec <- arrow::read_feather('corpora/cnec/named_ent_dtest.feather')

cnec <- recode_iob(cnec, colname = 'CoNLL_IOB2')
cnec$position <- as.numeric(cnec$position)
cnec$sentence <- as.numeric(gsub('s', '', cnec$sentence_id))
cnec$token_id <- NULL

# Transform raw corpus to tcorpus obect
tc_cnec <- corpustools::tokens_to_tcorpus(cnec, doc_col = 'sentence', token_id_col = 'position', token_col = 'token')

# Run Dictionary over Corpus
start_time <- Sys.time()
tc_cnec$code_dictionary(jrc_dict, case_sensitive = T, token_col = 'token', string_col = 'keyword', sep = ' ', use_wildcards = F)
end_time <- Sys.time()

# Recode Dictionary Results
codings_cnec <- recode_results(tc_cnec$tokens)

# Calculate Evaluation Metrics
cm_cnec <- caret::confusionMatrix(codings_cnec$JRC_NE, reference=codings_cnec$NE, mode = "everything")

r <- cm2df(cm_cnec, 'CNEC-DTest', 'cz')
r$validation_duration <- as.double(difftime(t2, t1, units="secs"))
results <- rbind(results, r)


# Save results

if (!dir.exists('results')) {
  dir.create('results')
}

write.csv(results, file='results/jrc_cnec.csv', row.names = F)
