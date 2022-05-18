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


corpora <- c('arabic_VALIDATION.feather', 'chinese_VALIDATION.feather', 'english_VALIDATION.feather')

corpora <- paste0('corpora/ontonotes/', corpora)

for (corpus in corpora) {
  
  # Load corpus
  df <- arrow::read_feather(corpus)
  
  df <- recode_iob(df, colname = 'CoNLL_IOB2')
  
  # Transform raw corpus to tcorpus obect
  tc <- corpustools::tokens_to_tcorpus(df, doc_col = 'doc_id', token_id_col = 'token_id', token_col = 'token')
  
  # Run Dictionary over Corpus
  start_time <- Sys.time()
  tc$code_dictionary(jrc_dict, case_sensitive = T, token_col = 'token', string_col = 'keyword', sep = ' ', use_wildcards = F)
  end_time <- Sys.time()
  
  # Recode Dictionary Results
  codings <- recode_results(tc$tokens)
  
  # Calculate Evaluation Metrics
  cm <- caret::confusionMatrix(codings$JRC_NE, reference=codings$NE, mode = "everything")
  
  r <- cm2df(cm, "ontonotes_validation", df$language[1])
  r$evaluation_time <-
    as.double(difftime(end_time, start_time, units = "secs"))
  
  results <- rbind(results, r)
  
}

# Save results

if (!dir.exists('results')) {
  dir.create('results')
}

write.csv(results, file='results/jrc_ontonotes.csv', row.names = F)
