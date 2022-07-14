library(arrow)
library(readr)
library(tidyverse)
library(dotwhisker)

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


training_combinations <- read_feather("tools/xlmroberta/training_combinations.feather")
training_combinations$model_id <- as.integer(row.names(training_combinations)) - 1
training_combinations <- training_combinations[, 3:5]
colnames(training_combinations) <- c('training_tokens', "training_sentences", "model_id")

df <- left_join(df, training_combinations, by="model_id")

# dummy variable to check whether the model was trained with the language it is validated against (= native)
# native = TRUE : model was trained on this language

df$native <- FALSE

for (i in 1:nrow(df)) {
  df$native[i] <- str_detect(df$model_languages[i], df$language[i])
}

df$native <- as.integer(df$native)

# dummy variable for noisy data

df$noisy <- as.integer(str_detect(df$model_corpora, "(emerging)|(europeana)"))

# some general insights

# which model performed the best on average
f1_mean <- df %>% filter(task == 'overall') %>% group_by(model_id) %>% summarise(mean(f1))

# how does corpora count affect f1 score (by language)?

df %>% 
  filter(task == 'overall') %>% 
  ggplot(aes(x=number_model_corpora, y=f1)) +
    geom_point() +
  geom_smooth() +
  facet_wrap(~language)



df %>% 
  filter(task == 'overall') %>% 
  ggplot(aes(x=number_model_languages, y=f1)) +
  geom_point() +
  geom_smooth() +
  facet_wrap(~language)


df %>% 
  filter(task == 'overall') %>% 
  ggplot(aes(x=language, y=f1)) +
  geom_point() +
  geom_boxplot() +
  facet_wrap(~as.factor(native))


df %>% 
  filter(task == 'overall') %>% 
  ggplot(aes(x=language, y=f1)) +
  geom_point() +
  geom_boxplot() +
  facet_wrap(~as.factor(noisy))



df_overall <- df  %>% 
  filter(task == 'overall') 

df_per <- df  %>% 
  filter(task == 'PER') 

m <- glm(f1 ~ number_model_corpora + number_model_languages + scale(training_tokens) + scale(training_sentences) + 
           `model_cnec2.0` + model_ontonotes + model_emerging + model_europeana + model_germeval2014 + model_conll + native + noisy, 
         data = df_overall)

summary(m)
dwplot(m)


m <- glm(f1 ~ number_model_corpora + number_model_languages + scale(training_tokens) + scale(training_sentences) + 
           + native + noisy, 
         data = df_overall)

summary(m)
dwplot(m) + theme_minimal()

regressions <- list()
i <- 1
for (l in unique(df_overall$language)) {
  m <- glm(f1 ~ number_model_corpora + number_model_languages + 
             `model_cnec2.0` + model_ontonotes + model_emerging + model_europeana + model_germeval2014 + model_conll + native + noisy, 
           data = df_overall[df_overall$language == l, ])
  regressions[[i]] <- m
  i <- i + 1
}

names(regressions) <- unique(df_overall$language)

dwplot(regressions, vline = geom_vline(
  xintercept = 0,
  colour = "grey60",
  linetype = 2)
) + theme_minimal()


regressions <- list()
i <- 1
for (l in unique(df_overall$language)) {
  m <- glm(f1 ~ number_model_corpora + number_model_languages + 
             native + noisy, 
           data = df_overall[df_overall$language == l, ])
  regressions[[i]] <- m
  i <- i + 1
}

names(regressions) <- unique(df_overall$language)

dwplot(regressions, vline = geom_vline(
  xintercept = 0,
  colour = "grey60",
  linetype = 2)
) + theme_minimal()


scale_range <- function(x){ (x-min(x))/(max(x)-min(x))}


regressions <- list()
i <- 1
for (l in unique(df_per$language)) {
  m <- glm(f1 ~ number_model_corpora + number_model_languages +
             `model_cnec2.0` + model_ontonotes + model_emerging + model_europeana + model_germeval2014 + model_conll + native + noisy, 
           data = df_per[df_per$language == l, ])
  regressions[[i]] <- m
  i <- i + 1
}

names(regressions) <- unique(df_per$language)

dwplot(regressions, vline = geom_vline(
  xintercept = 0,
  colour = "grey60",
  linetype = 2)
) + theme_minimal()


regressions <- list()
i <- 1
for (l in unique(df_per$language)) {
  m <- glm(f1 ~ number_model_corpora + number_model_languages +
             native + noisy, 
           data = df_per[df_per$language == l, ])
  regressions[[i]] <- m
  i <- i + 1
}

names(regressions) <- unique(df_per$language)

dwplot(regressions, vline = geom_vline(
  xintercept = 0,
  colour = "grey60",
  linetype = 2)
) + theme_minimal()


