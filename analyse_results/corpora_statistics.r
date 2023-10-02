library(readr)
library(dply)
library(ggplot2)
library(RColorBrewer)
library(showtext)
library(patchwork)
showtext_auto()
font_add_google("Source Sans Pro", "source")

if (!dir.exists('plots')) {
  dir.create('plots')
}

df <- read_csv("corpora/summary.csv")

df$`B-MISC`[is.na(df$`B-MISC`)] <- 0
df$`I-MISC`[is.na(df$`I-MISC`)] <- 0

df$check <- df$tokens == (df$O + df$`B-LOC` + df$`I-LOC` + 
                            df$`B-PER` + df$`I-PER` + 
                            df$`B-ORG` + df$`I-ORG` +
                            df$`B-MISC` + df$`I-MISC`)
View(df[is.na(df$check), ])

