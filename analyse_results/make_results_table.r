library(readr)
library(tidyverse)
library(RColorBrewer)
library(showtext)
library(patchwork)
library(viridis)

showtext_auto()
font_add_google("Source Sans Pro", "source")

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

# Rename Values and Variables ---------------------------------------------
source("analyse_results/utils.r")

# Make Corpora Pretty
results$corpus <- str_replace_all(results$corpus, pretty_corpora)
results$corpus <- factor(results$corpus, 
  levels = c("CoNLL", "AnCora", "HAREM", "AQMAR", 
             "OntoNotes", "SoNaR", "CNEC 2.0", 
             "GermEval", "Europeana", "Emerging Entities", 
             "WikiANN", "HIPE", "FiNER", "KIND", "NYTK-NerKor"))

# Make Languages Pretty
results$language <- str_replace_all(results$language, language_codes)

langs <- levels(as.factor(results$language))

# Make Tools pretty
results$tool <- str_replace_all(results$tool, tools_pretty)

results$Task <- str_replace_all(results$task, tasks_pretty)

results$Task <- factor(results$Task, 
  levels = c("Persons", "Organizations", "Locations",
  "Misc", "Overall", "O"))

# Plots for Precision / Recall / F1 ---------------------------------------

# Plot: Persons

results_average <- results %>% filter(task == 'PER') %>% 
  filter(language %in% langs) %>% 
  complete(tool, language) %>% 
  mutate(across(c(precision:f1), ~ifelse(is.na(.x), 0, .x))) %>% 
  group_by(tool, language) %>% 
  summarise(across(c(precision:f1), ~mean(.x, na.rm = T)))


results_average$prec_rec <- paste(round(results_average$precision * 100, 0), '/', round(results_average$recall * 100, 0))
results_average$f1 <- results_average$f1 * 100

blue <- rgb(0.23001121,	0.34597357,	0.60240251)

per <- ggplot(results_average, aes(language, tool)) +
  geom_tile(aes(fill = f1)) +
  geom_text(aes(label = prec_rec)) +
  scale_fill_gradient(low = "white", high = blue, name = expression(F[1]), limits = c(0,100)) +
  theme_minimal(base_family = "source", base_size = 18) +
  ggtitle("NER Task: Persons") +
  xlab("") +
  ylab("") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
  legend.position = "top")

per

ggsave('plots/results_persons.pdf', per, width = 14, height = 7, units = 'cm', scale = 2)
ggsave('plots/results_persons.png', per, width = 12, height = 9, units = 'cm', scale = 1, dpi=320)

# Plot: Organizations

results_average <- results %>% filter(task == 'ORG') %>% 
  filter(language %in% langs) %>% 
  complete(tool, language) %>% 
  mutate(across(c(precision:f1), ~ifelse(is.na(.x), 0, .x))) %>% 
  group_by(tool, language) %>% 
  summarise(across(c(precision:f1, validation_duration), ~mean(.x, na.rm = T)))

results_average$prec_rec <- paste(round(results_average$precision * 100, 0), '/', round(results_average$recall * 100, 0))
results_average$f1 <- results_average$f1 * 100

org <- ggplot(results_average, aes(language, tool)) +
  geom_tile(aes(fill = f1)) +
  geom_text(aes(label = prec_rec)) +
  scale_fill_gradient(low = "white", high = blue, name = expression(F[1]), limits = c(0,100)) +
  theme_minimal(base_family = "source", base_size = 18) +
  ggtitle("NER Task: Organizations") +
  xlab("") +
  ylab("") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
  legend.position = "top")

org

ggsave('plots/results_organizations.pdf', org, width = 14, height = 7, units = 'cm', scale = 2)
ggsave('plots/results_organizations.png', org, width = 12, height = 6, units = 'cm', scale = 1, dpi=320)

# Plot: Locations

results_average <- results %>% filter(task == 'LOC') %>% 
  filter(language %in% langs) %>% 
  complete(tool, language) %>% 
  mutate(across(c(precision:f1), ~ifelse(is.na(.x), 0, .x))) %>% 
  group_by(tool, language) %>% 
  summarise(across(c(precision:f1, validation_duration), ~mean(.x, na.rm = T)))

results_average$prec_rec <- paste(round(results_average$precision * 100, 0), '/', round(results_average$recall * 100, 0))
results_average$f1 <- results_average$f1 * 100

loc <- ggplot(results_average, aes(language, tool)) +
  geom_tile(aes(fill = f1)) +
  geom_text(aes(label = prec_rec)) +
  scale_fill_gradient(low = "white", high = blue, name = expression(F[1]), limits = c(0,100)) +
  theme_minimal(base_family = "source", base_size = 18) +
  ggtitle("NER Task: Locations") +
  xlab("") +
  ylab("") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
  legend.position = "top")

loc

ggsave('plots/results_locations.pdf', loc, width = 14, height = 7, units = 'cm', scale = 2)
ggsave('plots/results_locations.png', loc, width = 12, height = 6, units = 'cm', scale = 1, dpi=320)


# assemble into one figure

results_average <- results %>% 
  filter(language %in% langs) %>% 
  complete(Task, tool, language) %>% 
  mutate(across(c(precision:f1), ~ifelse(is.na(.x), 0, .x))) %>% 
  group_by(Task, tool, language) %>% 
  summarise(across(c(precision:f1), ~mean(.x, na.rm = T)))


results_average$prec_rec <- paste(round(results_average$precision * 100, 0), '/', round(results_average$recall * 100, 0))
results_average$f1 <- results_average$f1 * 100

p <- results_average |> 
  filter(Task %in% c("Persons", "Organizations", "Locations")) |> 
  ggplot(aes(language, tool)) +
    geom_tile(aes(fill = f1)) +
    geom_text(aes(label = prec_rec)) +
    scale_fill_gradient(low = "white", high = blue, name = expression(F[1]), limits = c(0,100)) +
    theme_minimal(base_family = "source", base_size = 18) +
    xlab("") +
    ylab("") +
    theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
      legend.position = "top",
      legend.justification = "right") +
    facet_wrap(~Task, ncol = 1)

ggsave('plots/results_languages.pdf', p, width = 14, height = 20, units = 'cm', scale = 2.3)
ggsave('plots/results_languages.png', p, width = 12, height = 18, units = 'cm', scale = 1, dpi=320)


# Precision / Recall: Performance by Corpus -------------------------------


results_average <- results %>% filter(task == 'PER') %>% 
  complete(tool, corpus) %>% 
  mutate(across(c(precision:f1), ~ifelse(is.na(.x), 0, .x))) %>% 
  group_by(tool, corpus) %>% 
  summarise(across(c(precision:f1), ~mean(.x, na.rm = T)))


results_average$prec_rec <- paste(round(results_average$precision * 100, 0), '/', round(results_average$recall * 100, 0))
results_average$f1 <- results_average$f1 * 100

per <- ggplot(results_average, aes(corpus, tool)) +
  geom_tile(aes(fill = f1)) +
  geom_text(aes(label = prec_rec)) +
  scale_fill_gradient(low = "white", high = blue, name = expression(F[1]), limits = c(0,100)) +
  theme_minimal(base_family = "source", base_size = 18) +
  ggtitle("NER Task: Persons") +
  xlab("") +
  ylab("") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
  legend.position = "top")

per


results_average <- results %>% filter(task == 'ORG') %>% 
  complete(tool, corpus) %>% 
  mutate(across(c(precision:f1), ~ifelse(is.na(.x), 0, .x))) %>% 
  group_by(tool, corpus) %>% 
  summarise(across(c(precision:f1), ~mean(.x, na.rm = T)))


results_average$prec_rec <- paste(round(results_average$precision * 100, 0), '/', round(results_average$recall * 100, 0))
results_average$f1 <- results_average$f1 * 100

org <- ggplot(results_average, aes(corpus, tool)) +
  geom_tile(aes(fill = f1)) +
  geom_text(aes(label = prec_rec)) +
  scale_fill_gradient(low = "white", high = blue, name = expression(F[1]), limits = c(0,100)) +
  theme_minimal(base_family = "source", base_size = 18) +
  ggtitle("NER Task: Organizations") +
  xlab("") +
  ylab("") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
  legend.position = "top")

org


results_average <- results %>% filter(task == 'LOC') %>% 
  complete(tool, corpus) %>% 
  mutate(across(c(precision:f1), ~ifelse(is.na(.x), 0, .x))) %>% 
  group_by(tool, corpus) %>% 
  summarise(across(c(precision:f1), ~mean(.x, na.rm = T)))


results_average$prec_rec <- paste(round(results_average$precision * 100, 0), '/', round(results_average$recall * 100, 0))
results_average$f1 <- results_average$f1 * 100

loc <- ggplot(results_average, aes(corpus, tool)) +
  geom_tile(aes(fill = f1)) +
  geom_text(aes(label = prec_rec)) +
  scale_fill_gradient(low = "white", high = blue, name = expression(F[1]), limits = c(0,100)) +
  theme_minimal(base_family = "source", base_size = 18) +
  ggtitle("NER Task: Locations") +
  xlab("") +
  ylab("") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
  legend.position = "top")

loc


# assemble into one figure

results_average <- results %>% 
  complete(Task, tool, corpus) %>% 
  mutate(across(c(precision:f1), ~ifelse(is.na(.x), 0, .x))) %>% 
  group_by(Task, tool, corpus) %>% 
  summarise(across(c(precision:f1), ~mean(.x, na.rm = T)))


results_average$prec_rec <- paste(round(results_average$precision * 100, 0), '/', round(results_average$recall * 100, 0))
results_average$f1 <- results_average$f1 * 100

p <- results_average |> 
  filter(Task %in% c("Persons", "Organizations", "Locations")) |> 
  ggplot(aes(corpus, tool)) +
    geom_tile(aes(fill = f1)) +
    geom_text(aes(label = prec_rec)) +
    scale_fill_gradient(low = "white", high = blue, name = expression(F[1]), limits = c(0,100)) +
    theme_minimal(base_family = "source", base_size = 18) +
    xlab("") +
    ylab("") +
    theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5),
      legend.position = "top",
      legend.justification = "right") +
    facet_wrap(~Task, ncol = 1)

ggsave('plots/results_corpora.pdf', p, width = 15, height = 20, units = 'cm', scale = 2.3)
ggsave('plots/results_corpora.png', p, width = 12, height = 18, units = 'cm', scale = 1, dpi=320)