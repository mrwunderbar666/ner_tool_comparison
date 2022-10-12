library(readr)
library(tidyverse)
library(RColorBrewer)
library(showtext)
library(patchwork)
showtext_auto()
font_add_google("Source Sans Pro", "source")

if (!dir.exists('plots')) {
  dir.create('plots')
}


# Load Corpus Registry ----------------------------------------------------
# Helps to compare corpus sizes

corpora <- read_csv("corpora/registry.csv")
corpora <- corpora %>% filter(split == 'validation')


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

per <- ggplot(results_average, aes(language, tool)) +
  geom_tile(aes(fill = f1)) +
  geom_text(aes(label = prec_rec)) +
  scale_fill_gradient(low = "white", high = "#0063A6", name = expression(F[1]), limits = c(0,100)) +
  theme_minimal(base_family = "source", base_size = 18) +
  ggtitle("NER Task: Persons") +
  xlab("") +
  ylab("") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))

per

ggsave('plots/results_persons.pdf', per, width = 12, height = 6, units = 'cm', scale = 2)
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
  scale_fill_gradient(low = "white", high = "#0063A6", name = expression(F[1]), limits = c(0,100)) +
  theme_minimal(base_family = "source", base_size = 18) +
  ggtitle("NER Task: Organizations") +
  xlab("") +
  ylab("") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))

org

ggsave('plots/results_organizations.pdf', org, width = 12, height = 6, units = 'cm', scale = 2)
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
  scale_fill_gradient(low = "white", high = "#0063A6", name = expression(F[1]), limits = c(0,100)) +
  theme_minimal(base_family = "source", base_size = 18) +
  ggtitle("NER Task: Locations") +
  xlab("") +
  ylab("") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))

loc

ggsave('plots/results_locations.pdf', loc, width = 12, height = 6, units = 'cm', scale = 2)
ggsave('plots/results_locations.png', loc, width = 12, height = 6, units = 'cm', scale = 1, dpi=320)


# assemble into one figure

p <- per / org / loc + plot_annotation(title = "Average precision / recall across all corpora", theme = theme_minimal(base_family = "source", base_size = 18))

ggsave('plots/results_languages.pdf', p, width = 12, height = 18, units = 'cm', scale = 2.3)
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
  scale_fill_gradient(low = "white", high = "#0063A6", name = expression(F[1]), limits = c(0,100)) +
  theme_minimal(base_family = "source", base_size = 18) +
  ggtitle("NER Task: Persons") +
  xlab("") +
  ylab("") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))

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
  scale_fill_gradient(low = "white", high = "#0063A6", name = expression(F[1]), limits = c(0,100)) +
  theme_minimal(base_family = "source", base_size = 18) +
  ggtitle("NER Task: Organizations") +
  xlab("") +
  ylab("") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))

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
  scale_fill_gradient(low = "white", high = "#0063A6", name = expression(F[1]), limits = c(0,100)) +
  theme_minimal(base_family = "source", base_size = 18) +
  ggtitle("NER Task: Locations") +
  xlab("") +
  ylab("") +
  theme(axis.text.x = element_text(angle = 90, hjust = 1, vjust = 0.5))

loc


# assemble into one figure

p <- per / org / loc + 
  plot_annotation(title = "Average precision / recall across all languages", 
                  theme = theme_minimal(base_family = "source", base_size = 18))


ggsave('plots/results_corpora.pdf', p, width = 12, height = 18, units = 'cm', scale = 2.3)
ggsave('plots/results_corpora.png', p, width = 12, height = 18, units = 'cm', scale = 1, dpi=320)



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