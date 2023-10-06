library(corpustools)
library(readr)
library(arrow)
library(stringi)
library(caret)

args <-  commandArgs(trailingOnly=TRUE)

if ('--debug' %in% args) {
  print('debug mode')
  debug <- TRUE
} else {
  debug <- FALSE
}

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
jrc_dict$keyword <- stringi::stri_replace_all(jrc_dict$keyword, fixed = '+', ' ')

registry <- read_csv('corpora/registry.csv')
registry <- registry[registry$split == 'validation', ]

languages <- c("en", "de", 'nl', 'es', 'fr', 'cs', 'hu', 'it', 'ar', 'zh')

corpora <- registry[registry$language %in% languages, ]

for (i in 1:nrow(corpora)) {
  print(corpora$path[i])
  
  # Load Data
  df <- arrow::read_feather(corpora$path[i])
  
  if ('doc_id' %in% colnames(df)) {
    df$doc_id <- NULL
  }

  if (debug) {
    sentence_ids <- unique(df$sentence_id)
    sample_size <- min(c(length(sentence_ids), 100))
    random_sentences <- sample(sentence_ids, sample_size)
    filt <- df$sentence_id %in% random_sentences
    df <- df[filt, ]
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
  
  r <- cm2df(cm, corpora$corpus[i], corpora$language[i])
  r$evaluation_time <-
    as.double(difftime(end_time, start_time, units = "secs"))
  
  r$corpus <- corpora$corpus[i]
  r$subset <- corpora$subset[i]
  r$language <- corpora$language[i]
  r$sentences <- corpora$sentences[i]
  r$tokens <- corpora$tokens[i]
  
  results <- rbind(results, r)
  
}

# Save results

if (!dir.exists('results')) {
  dir.create('results')
}

write.csv(results, file = 'results/jrc.csv', row.names = F)

