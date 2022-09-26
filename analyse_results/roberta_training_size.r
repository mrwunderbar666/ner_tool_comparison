library(readr)
library(arrow)
library(tidyverse)
library(RColorBrewer)
library(showtext)
showtext_auto()
font_add_google("Source Sans Pro", "source")

df <- read_feather("results/roberta_training_varying_size.feather")

df$`Training Samples` <- df$training_sentences
df$`F1` <- df$`results.eval_f1`
df$`precision` <- df$`results.eval_precision`
df$`recall` <- df$`results.eval_recall`

p <- df %>% ggplot(aes(x=`Training Samples`, y=`F1`)) +
  geom_line(size=1) +
  geom_point(size=3) +
  # scale_x_continuous(minor_breaks=seq(500, 6500, 500), breaks = seq(0, 7000, 1000), limits = c(0, 49000)) +
  scale_x_log10() +
  scale_y_continuous(breaks = seq(0, 1, 0.2), limits = c(0, 1.0)) +
  theme_minimal(base_family = "source", base_size = 18) +
  ggtitle("XLM-RoBERTa multilingual training", subtitle = "logarithmic scaled x-axis.")

p

ggsave('plots/roberta_training.pdf', p, width = 12, height = 6, units = 'cm', scale = 2)
ggsave('plots/roberta_training.png', p, width = 12, height = 5, units = 'cm', scale = 1, dpi = 320)