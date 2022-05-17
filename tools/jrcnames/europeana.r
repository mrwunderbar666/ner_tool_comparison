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

corpora <- Sys.glob('corpora/europeana/*_validation.feather')

for (corpus in corpora) {
  
  print(corpus)
  
  # Load corpus
  df <- arrow::read_feather(corpus)
  
  df <- recode_iob(df, colname = 'IOB2')
  
  # Transform raw corpus to tcorpus obect
  tc <- corpustools::tokens_to_tcorpus(df, doc_col = 'sentence', token_id_col = 'token_id', token_col = 'token')
  
  # Run Dictionary over Corpus
  tc$code_dictionary(jrc_dict, case_sensitive = T, token_col = 'token', string_col = 'keyword', sep = ' ', use_wildcards = F)
  
  # Recode Dictionary Results
  codings <- recode_results(tc$tokens)
  
  # Calculate Evaluation Metrics
  cm <- caret::confusionMatrix(codings$JRC_NE, reference=codings$NE, mode = "everything")
  
  r <- cm2df(cm, df$corpus[1], df$language[1])
  
  results <- rbind(results, r)
  
  cm <- caret::confusionMatrix(as.factor(codings$JRC_any_NE), reference=as.factor(codings$any_NE), mode = "everything")
  
  results <- rbind(results, data.frame(
    corpus = df$corpus[1],
    lang = df$language[1], 
    task = 'any_NE', 
    precision = cm$byClass['Precision'],
    recall = cm$byClass['Recall'],
    specificity = cm$byClass['Specificity'],
    f1 = cm$byClass['F1']
  ))
# 
}

# Save results

if (!dir.exists('results')) {
  dir.create('results')
}

write.csv(results, file='results/jrc_europeana.csv', row.names = F)
