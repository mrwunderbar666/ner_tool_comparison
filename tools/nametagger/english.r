library(nametagger)
library(arrow)
library(stringr)
library(readr)

if (!dir.exists("tools/nametagger/tmp")) {
  dir.create("tools/nametagger/tmp")
}

if (!file.exists("tools/nametagger/tmp/english-conll-140408.ner")) { 
  model <- nametagger_download_model(language = "english-conll-140408", model_dir = "tools/nametagger/tmp")
} else {
  model <- nametagger_load_model("tools/nametagger/tmp/english-conll-140408.ner")
}

language <- "en"

registry <- read_csv("corpora/registry.csv")
registry <- registry[registry$split == 'validation', ]

corpora <- registry[registry$language %in% language, ]

for (i in 1:nrow(corpora)) {
  
  print(corpora$path[i])
  
  corpus <- arrow::read_feather(corpora$path[i])
  
  # satisfy the expected input of nametagger
  
  corpus$text <- corpus$token
  
  if (!"doc_id" %in% colnames(corpus)) {
    corpus$doc_id <- "1"
  }
  
  
  if (!"sentence_id" %in% colnames(corpus)) {
    corpus$sentence_id <- corpus$doc_id
  }
  
  if (!is.numeric(corpus$sentence_id)) {
    if (length(which(stringr::str_detect(corpus$sentence_id, '_'))) > 0) {
      corpus$sentence_id <- gsub("_", '', corpus$sentence_id, fixed = T)
    }
    corpus$sentence_id <- as.numeric(corpus$sentence_id)
    
  }
  
  corpus$CoNLL_IOB2 <- as.factor(corpus$CoNLL_IOB2)
  
  start_time <- Sys.time()
  annotations <- predict(model, corpus)
  end_time <- Sys.time()
  
  predictions <- data.frame(
    sentence_id = corpus$sentence_id,
    nametagger = factor(annotations$entity, levels = levels(corpus$CoNLL_IOB2)),
    references = corpus$CoNLL_IOB2
  )
  
  write_feather(predictions, paste0("tools/nametagger/tmp/", corpora$corpus[i], '_', corpora$subset[i], '_', language, '.feather'))
  runtime <- data.frame(
            corpus = corpora$corpus[i],
            subset = corpora$subset[i],
            language = language,
            sentences = corpora$sentences[i],
            tokens = corpora$tokens[i],
            evaluation_time=as.double(difftime(end_time, start_time, units = "secs"))
  )
  write_csv(runtime, paste0("tools/nametagger/tmp/", corpora$corpus[i], '_', corpora$subset[i], '_', language, '.csv'))
            
  }
