
for (rdata in Sys.glob('tools/icews/*.RData')) {
  load(rdata)
}

icews_actors <- x[, c('Actor.Name', 'Actor.Type', 'Aliases')]
icews_actors <- unique(icews_actors)
icews_actors$Aliases <- paste0(icews_actors$Actor.Name, ' || ', icews_actors$Aliases)

library(data.table)

dt <- as.data.table(icews_actors)

DT = data.table(read.table(header=T, text="blah | splitme
    T | a,b,c
    T | a,c
    F | b,d
    F | e,f", stringsAsFactors=F, sep="|", strip.white = TRUE))

dt <- dt[,.( Actor.Name
       , Actor.Type
       , Aliases=unlist(strsplit(Aliases, " || ", fixed=TRUE))
),by=seq_len(nrow(dt))]
