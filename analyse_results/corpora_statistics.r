library(readr)
library(dplyr)
library(tidyr)
library(stringr)
library(ggplot2)
library(RColorBrewer)
library(showtext)
library(patchwork)
library(kableExtra)
library(viridis)

showtext_auto()
font_add_google("Source Sans Pro", "source")

if (!dir.exists('plots')) {
  dir.create('plots')
}

source("analyse_results/utils.r")

df <- read_csv("corpora/summary.csv")

df$`B-MISC`[is.na(df$`B-MISC`)] <- 0
df$`I-MISC`[is.na(df$`I-MISC`)] <- 0

df$check <- df$tokens == (df$O + df$`B-LOC` + df$`I-LOC` + 
                            df$`B-PER` + df$`I-PER` + 
                            df$`B-ORG` + df$`I-ORG` +
                            df$`B-MISC` + df$`I-MISC`)
View(df[is.na(df$check), ])

languages <- unique(df[df$corpus != "wikiann", ]$language)

df$`Released` <- str_replace_all(df$corpus, corpus_release_year)

# Big Picture of Validation Datasets

corpus_summary_table <- df |> filter(language %in% languages) |> 
  mutate(Corpus = str_replace_all(corpus, pretty_corpora)) |> 
  group_by(Corpus) |> 
  mutate(Languages = paste(
    str_sort(unique(
      str_replace_all(language, language_codes))), 
      collapse = ", ")) |> 
  filter(split == 'validation') |> 
  group_by(Corpus, Released, Languages) |> 
  summarise(Sentences = sum(sentences), Tokens = sum(tokens)) |> 
  arrange(Released)

# latex export
knitr::kable(corpus_summary_table, "latex", 
              format.args = list(big.mark = ','),
              booktabs = TRUE)


# Breakdown of NE Categories per corpus

ne_categories <- df |> filter(language %in% languages) |> 
  mutate(Tokens = tokens,
      Persons = `B-PER` + `I-PER`,
      Organizations = `B-ORG` + `I-ORG`,
      Locations = `B-LOC` + `I-LOC`,
      Misc = `B-MISC` + `I-MISC`) |> 
  mutate(Corpus = str_replace_all(corpus, pretty_corpora), 
         Language = str_replace_all(language, language_codes)) |> 
  group_by(Corpus, Released, Language) |> 
  summarise(across(Tokens:Misc, sum)) |> 
  mutate(across(Persons:Misc, ~ round((.x / Tokens)*100, 2), .names = "{.col} %" )) |> 
  select(Corpus:Tokens, 
         starts_with("Pers"),
         starts_with("Org"),
         starts_with("Loc"),
         starts_with("Misc")
         ) |> 
  arrange(Released)

ne_categories_mean <- ne_categories |> 
                        ungroup() |> 
                        summarise(across(Tokens:last_col(), ~ mean(.x, na.rm = TRUE))) |> 
                        mutate(across(!ends_with("%"), as.integer)) |> 
                        mutate(Corpus = 'mean', Released = "", Language = "")



ne_categories |> bind_rows(ne_categories_mean) |> 
                      knitr::kable("latex",
                              format.args = list(big.mark = ','),
                              booktabs = TRUE,
                              digits = 2,
                              col.names = c("Corpus",
                                            "Released",
                                            "Language",
                                            "Tokens",
                                            "N", "%",
                                            "N", "%",
                                            "N", "%",
                                            "N", "%"
                                            )) |>
      add_header_above(c(" ", " ", " ", " ", "Persons" = 2, "Organizations" = 2, "Locations" = 2, "Misc" = 2))


# Gender Bias in Corpora

gender_bias_corpora <- df |> filter(language %in% languages) |> 
  filter(language != 'ar') |> 
  mutate(Corpus = str_replace_all(corpus, pretty_corpora), 
         Language = str_replace_all(language, language_codes)) |> 
  group_by(Corpus, Language) |> 
  summarise(across(`NA`:F, sum)) |> 
  mutate(Persons = M + F + `NA`) |>
  mutate(Male = M / Persons, Female = F / Persons, Unknown = (`NA` / Persons) / 2) |> 
  select(Corpus, Language, Male, Female, Unknown) |> 
  pivot_longer(Male:Unknown, values_to = "Share", names_to = "Gender") |> 
  drop_na()

male <- gender_bias_corpora |> filter(Gender != 'Female')

female <- gender_bias_corpora |> filter(Gender != 'Male') |> 
  mutate(Share = -Share)

gender_plot_corpora <- ggplot() +
    geom_col(data=male,
              aes(y=Language, x=Share, fill=Gender), 
              position = position_stack()) +
    geom_col(data=female,
              aes(y=Language, x=Share, fill=Gender),  
              position = position_stack()) +
    geom_vline(xintercept = 0, linetype = "dotted") +
    scale_x_continuous(
      breaks = c(
          pretty(male$Share), 
          pretty(female$Share)
          ),
      labels = abs(
        c(
          pretty(male$Share), 
          pretty(female$Share)
          )
          )) +
    scale_fill_viridis(discrete = TRUE, option = "mako",
                        begin = 0.3, end = 0.7) +
    theme_minimal(base_family = "source", base_size = 18) +
    theme(legend.position = "top") +
    facet_wrap(~Corpus, ncol = 3)


ggsave('plots/corpora_gender_faceted.pdf', gender_plot_corpora, width = 21, height = 29, units = 'cm', scale = 2)

# Gender Bias Overall for Languages

gender_bias_language <- df |> filter(language %in% languages) |> 
  filter(language != 'ar') |> 
  mutate(Language = str_replace_all(language, language_codes)) |> 
  group_by(Language) |> 
  summarise(across(`NA`:F, sum)) |> 
  mutate(Persons = M + F + `NA`) |>
  mutate(Male = M / Persons, Female = F / Persons, Unknown = (`NA` / Persons) / 2) |> 
  select(Language, Male, Female, Unknown) |> 
  pivot_longer(Male:Unknown, values_to = "Share", names_to = "Gender") |> 
  drop_na()

male <- gender_bias_language |> filter(Gender != 'Female')

female <- gender_bias_language |> filter(Gender != 'Male') |> 
  mutate(Share = -Share)

gender_plot_languages <- ggplot() +
    geom_col(data=male,
              aes(y=Language, x=Share, fill=Gender), 
              position = position_stack()) +
    geom_col(data=female,
              aes(y=Language, x=Share, fill=Gender),  
              position = position_stack()) +
    geom_vline(xintercept = 0, linetype = "dotted") +
    scale_x_continuous(
      breaks = c(
          pretty(male$Share), 
          pretty(female$Share)
          ),
      labels = abs(
        c(
          pretty(male$Share), 
          pretty(female$Share)
          )
          )) +
    scale_fill_viridis(discrete = TRUE, option = "mako",
                        begin = 0.3, end = 0.7) +
    theme_minimal(base_family = "source", base_size = 18) +
    theme(legend.position = "top")

ggsave('plots/languages_gender.pdf', gender_plot_languages, width = 12, height = 6, units = 'cm', scale = 2)
