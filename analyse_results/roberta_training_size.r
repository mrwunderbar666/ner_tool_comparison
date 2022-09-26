library(readr)
library(tidyverse)
library(RColorBrewer)
library(showtext)
showtext_auto()
font_add_google("Source Sans Pro", "source")

df <- read_csv("roberta_training_size.csv")

df$`Training Samples` <- df$training_size
df$`F1` <- df$`eval/f1`

p <- df %>% ggplot(aes(x=`Training Samples`, y=`F1`)) +
  geom_line(size=1) +
  geom_point(size=3) +
  scale_x_continuous(minor_breaks=seq(500, 6500, 500), breaks = seq(0, 7000, 1000), limits = c(0, 7000)) +
  scale_y_continuous(breaks = seq(0, 1, 0.2), limits = c(0, 1.0)) +
  theme_minimal(base_family = "source", base_size = 18) +
  ggtitle("XLM-RoBERTa multilingual training")


ggsave('roberta_training.pdf', p, width = 16, height = 9, units = 'cm', scale = 2)
ggsave('roberta_training.png', p, width = 16, height = 9, units = 'cm', scale = 1, dpi = 320)
