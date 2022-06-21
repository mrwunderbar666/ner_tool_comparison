library(readr)
library(dataverse)
library(data.table)

icews_actors <- get_dataframe_by_doi("10.7910/DVN/28118/HYSJN6",
                                      server = "dataverse.harvard.edu")

icews_actors <- icews_actors[, c('Actor Name', 'Actor Type', 'Aliases')]
icews_actors <- unique(icews_actors)
icews_actors$Aliases <- paste0(icews_actors$`Actor Name`, ' || ', icews_actors$Aliases)

dt <- as.data.table(icews_actors)

dt <- dt[,.(`Actor Name`, 
            `Actor Type`, 
            Aliases=unlist(strsplit(Aliases, " || ", fixed=TRUE))
            ),
         by=seq_len(nrow(dt))]


df <- as.data.frame(dt)

df <- unique(df)

df$id <- as.character(df$seq_len)
df$keyword <- trimws(as.character(df$Aliases))
df$type <- trimws(as.character(df$`Actor Type`))

df <- df[, c("id", "type", "keyword")]

df$type <- gsub('Individual', 'P', df$type, fixed = T)
df$type <- gsub('Group', 'O', df$type, fixed = T)

write_rds(df, 'tools/icews/icews_actors.rds')
