library(readr)
library(tidyverse)

files <- Sys.glob("tools/xlmroberta/models/*/*.csv")


eval_results <- read_csv(files[1])

for (f in files[2:length(files)]) {
  tmp <- read_csv(f)
  eval_results <- rbind(eval_results, tmp)
}

registry <- read_csv('corpora/registry.csv')
df <- left_join(eval_results, registry, by=c("validation_corpus"="path"))

# recode 
df$number_model_languages <-str_count(df$model_languages, ',')
df$number_model_languages <- df$number_model_languages + 1
df$number_model_corpora <-str_count(df$model_corpora, ',')
df$number_model_corpora <- df$number_model_corpora + 1

corpora_names <- unique(df$corpus)

for (i in corpora_names) {
  df[, paste0('model_', i)] <- as.integer(str_detect(df$model_corpora, i))
}

best_performance <- df %>% 
  group_by(task, validation_corpus) %>% 
  filter(f1 == max(f1)) %>% 
  distinct(validation_corpus, .keep_all = T)

best_performance$tool <- "xlm-roberta"

export_cols <- c("task", "precision", "recall", "f1", "number", "accuracy", "language", "corpus", "subset", "validation_duration", "tokens", "sentences", "tool")

write_csv(best_performance[, export_cols], 'results/xlmroberta.csv')
