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
jrc_dict$keyword <-
  stringi::stri_replace_all(jrc_dict$keyword, fixed = '+', ' ')

corpora <-
  c("esp.testb.feather",
    "ned.testb.feather",
    "conll2003_en_test_iob.feather")

corpora <- paste0("corpora/conll/", corpora)

for (corpus in corpora) {
  print(corpus)
  
  # Load Data
  df <- arrow::read_feather(corpus)
  
  if ("doc" %in% colnames(df)) {
    df$sentence_id <- paste(df$doc, df$sentence_id, sep = '_')
  }
  
  df <- recode_iob(df)
  
  # Transform raw corpus to tcorpus obect
  tc <-
    corpustools::tokens_to_tcorpus(df,
                                   doc_col = 'sentence_id',
                                   token_id_col = 'token_id',
                                   token_col = 'token')
  
  # Run Dictionary over Corpus
  start_time <- Sys.time()
  tc$code_dictionary(
    jrc_dict,
    case_sensitive = T,
    token_col = 'token',
    string_col = 'keyword',
    sep = ' ',
    use_wildcards = F
  )
  end_time <- Sys.time()
  
  # Recode Dictionary Results
  codings <- recode_results(tc$tokens)
  
  
  # Calculate Evaluation Metrics
  cm <-
    caret::confusionMatrix(codings$JRC_NE, reference = codings$NE, mode = "everything")
  
  r <- cm2df(cm, corpus, df$language[1])
  r$evaluation_time <-
    as.double(difftime(end_time, start_time, units = "secs"))
  
  results <- rbind(results, r)
  
}

# Save results

if (!dir.exists('results')) {
  dir.create('results')
}

write.csv(results, file = 'results/jrc_conll2002.csv', row.names = F)
