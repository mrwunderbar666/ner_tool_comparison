library(readr)
library(arrow)
library(curl)
library(nametagger)
library(dplyr)

args <- commandArgs(trailingOnly = TRUE)

if ("--debug" %in% args) {
  print("debug mode")
  debug <- TRUE
} else {
  debug <- FALSE
}

if (!dir.exists("tools/nametagger/tmp")) {
  dir.create("tools/nametagger/tmp")
}


if (!file.exists("tools/nametagger/tmp/czech-cnec-140304/czech-cnec2.0-140304-no_numbers.ner")) {
  curl_download(
    "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11858/00-097C-0000-0023-7D42-8/czech-cnec-140304.zip",
    "tools/nametagger/tmp/czech-cnec-140304.zip"
  )

  unzip("tools/nametagger/tmp/czech-cnec-140304.zip",
    exdir = "tools/nametagger/tmp"
  )
}

model <- nametagger_load_model("tools/nametagger/tmp/czech-cnec-140304/czech-cnec2.0-140304-no_numbers.ner")

language <- "cs"


registry <- read_csv("corpora/registry.csv")
registry <- registry[registry$split == "validation", ]

corpora <- registry[registry$language %in% language, ]


for (i in 1:nrow(corpora)) {
  print(corpora$path[i])

  corpus <- arrow::read_feather(corpora$path[i])

  if (debug) {
    sentence_ids <- unique(corpus$sentence_id)
    sample_size <- min(c(length(sentence_ids), 100))
    random_sentences <- sample(sentence_ids, sample_size)
    filt <- corpus$sentence_id %in% random_sentences
    corpus <- corpus[filt, ]
  }

  # satisfy the expected input of nametagger

  corpus$text <- corpus$token

  if (!"doc_id" %in% colnames(corpus)) {
    corpus$doc_id <- "1"
  }


  if (!"sentence_id" %in% colnames(corpus)) {
    corpus$sentence_id <- corpus$doc_id
  }

  if (!is.numeric(corpus$sentence_id)) {
    corpus$sentence_id <- as.numeric(as.factor(corpus$sentence_id))
  }

  x <- corpus |> group_by(doc_id, sentence_id) |> 
    mutate(text = paste(token, collapse = "\n")) |> 
    distinct(doc_id, sentence_id, .keep_all=T)

  start_time <- Sys.time()
  annotations <- predict(model, x)
  end_time <- Sys.time()

  predictions <- data.frame(
    sentence_id = corpus$sentence_id,
    nametagger = annotations$entity,
    references = corpus$CoNLL_IOB2
  )

  write_feather(predictions, paste0("tools/nametagger/tmp/", corpora$corpus[i], "_", corpora$subset[i], "_", language, ".feather"))
  runtime <- data.frame(
    corpus = corpora$corpus[i],
    subset = corpora$subset[i],
    language = language,
    sentences = corpora$sentences[i],
    tokens = corpora$tokens[i],
    evaluation_time = as.double(difftime(end_time, start_time, units = "secs"))
  )
  write_csv(runtime, paste0("tools/nametagger/tmp/", corpora$corpus[i], "_", corpora$subset[i], "_", language, ".csv"))
}
