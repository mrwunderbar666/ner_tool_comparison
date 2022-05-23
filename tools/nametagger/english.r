library(nametagger)
library(arrow)

if (!dir.exists("tools/nametagger/tmp")) {
  dir.create("tools/nametagger/tmp")
}

model <- nametagger_download_model(language = "english-conll-140408", model_dir = "tools/nametagger/tmp")

language <- "en"

corpora <- c(conll = "conll/conll2003_en_validation_iob.feather",
             emerging = "emerging/emerging.test.annotated.feather",
             ontonotes = "ontonotes/english_VALIDATION.feather",
             wikiann = "wikiann/wikiann-en_validation.feather")


for (c in corpora) {
  
  print(c)
  
  corpus <- arrow::read_feather(paste0("corpora/", c))
  
  # satisfy the expected input of nametagger
  
  corpus$text <- corpus$token
  
  if ('CoNLL_IOB2' %in% colnames(corpus)) {
    corpus$IOB2 <- corpus$CoNLL_IOB2
  }
  
  if (!"doc_id" %in% colnames(corpus)) {
    corpus$doc_id = '1'
  }
  
  if (!"sentence_id" %in% colnames(corpus)) {
    corpus$sentence_id = corpus$doc_id
  }
  
  corpus$IOB2 <- as.factor(corpus$IOB2)
  
  annotations <- predict(model, corpus)
  predictions <- data.frame(
    sentence_id = corpus$sentence_id,
    nametagger = factor(annotations$entity, levels = levels(corpus$IOB2)),
    references = corpus$IOB2
  )
  
  write_feather(predictions, paste0("tools/nametagger/tmp/", strsplit(c, '/', fixed=T)[[1]][1], '_', language, '.feather'))

}
