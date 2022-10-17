library(readr)
library(tidyverse)

if (!dir.exists('plots')) {
  dir.create('plots')
}


# Load Results from Tool Evaluations --------------------------------------

results <- data.frame(corpus=NULL, tool=NULL, task=NULL, language=NULL, precision=NULL, recall=NULL,
                      f1=NULL, validation_duration=NULL, training_duration=NULL)

for (f in Sys.glob('results/*.csv')) {
  
  tmp <- read_csv(f)
  tmp$tool <- strsplit(f, "_", fixed=T)[[1]][1]
  tmp$tool <- gsub("results/", "", tmp$tool)
  tmp$tool <- gsub(".csv", "", tmp$tool)
  
  if ("evaluation_time" %in% colnames(tmp)) {
    tmp$validation_duration <- tmp$evaluation_time
    tmp$evaluation_time <- NULL
  }
  
  if ("lang" %in% colnames(tmp)) {
    tmp$language <- tmp$lang
    tmp$lang <- NULL
  }
  
  if ("specificity" %in% colnames(tmp)) {
    tmp$specificity <- NULL
  }
  
  if ("validation_corpus" %in% colnames(tmp)) {
    tmp$validation_corpus <- NULL
  }
  
  if ("model_id" %in% colnames(tmp)) {
    tmp$model_id <- NULL
  }
  
  if ("model_languages" %in% colnames(tmp)) {
    tmp$model_languages <- NULL
  }
  
  if ("model_corpora" %in% colnames(tmp)) {
    tmp$model_corpora <- NULL
  }
  
  results <- dplyr::bind_rows(results, tmp)
  
}


# Recode spaCy ------------------------------------------------------------
# distinguish between transformer based models and "large" models

results[is.na(results$model), 'model'] <- ""
results[results$model == 'xx_ent_wiki_sm', 'tool'] <- "spacy-multi"

results$model_kind <- str_split_fixed(results$model, "_", n = 4)[, 4]

filt <- results$tool == 'spacy'

results[filt, 'tool'] <- paste(results[filt, 'tool'], results[filt, 'model_kind'], sep="_")

# Recode JRC Names and ICEWS


results[results$tool %in% c('jrc', 'icews') & results$task == 'O', 'task'] <- 'overall'

# Rename Values and Variables ---------------------------------------------
# Make Corpora Pretty


pretty_corpora <- c("conll" = "CoNLL", 
                    "ontonotes" = "OntoNotes", 
                    "germeval2014" = "GermEval", 
                    "cnec2.0" = "CNEC 2.0", 
                    "europeana" = "Europeana", 
                    "emerging" =  "Emerging Entities", 
                    "wikiann" = "WikiANN")

results$corpus <- str_replace_all(results$corpus, pretty_corpora)
results$corpus <- factor(results$corpus, levels = c("CoNLL", "OntoNotes",  "GermEval",  "CNEC 2.0",  "Europeana",  "Emerging Entities", "WikiANN"))

# Make Languages Pretty

language_codes <- c("ar" = "Arabic",
                    "es" = "Spanish",
                    "cs" = "Czech",
                    "zh" = "Chinese",
                    "nl" = "Dutch",
                    "en" = "English",
                    "fr" = "French",
                    "de" = "German", 
                    "hu" = "Hungarian",
                    "it" = "Italian"
)


results$language <- str_replace_all(results$language, language_codes)

langs <- levels(as.factor(results$language))

# Make Tools pretty

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


results$tool <- str_replace_all(results$tool, tools_pretty)


# Calculate Speed of Tools ------------------------------------------------
# in tokens per second

results$speed <- round(results$tokens / results$validation_duration)


results_speed <- results %>% filter(task == 'overall') %>% group_by(tool) %>% 
  summarise(`tokens / sec` = round(mean(speed), 0)) %>% 
  arrange(desc(`tokens / sec`))

# Pretty print numbers
results_speed$`tokens / sec` <- format(results_speed$`tokens / sec`, big.mark = ',', big.interval = 3L, justify = "none")

# export results into simple table
write_csv(results_speed, 'plots/speed_comparison.csv')

