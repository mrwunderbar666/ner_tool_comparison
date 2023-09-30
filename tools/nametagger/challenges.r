library(nametagger)
library(stringr)
library(jsonlite)
library(reticulate)

if (!dir.exists("tools/nametagger/tmp")) {
  dir.create("tools/nametagger/tmp")
}

if (!file.exists("tools/nametagger/tmp/english-conll-140408.ner")) { 
  model <- nametagger_download_model(language = "english-conll-140408", model_dir = "tools/nametagger/tmp")
} else {
  model <- nametagger_load_model("tools/nametagger/tmp/english-conll-140408.ner")
}

language <- "en"

reticulate::use_virtualenv(paste0(getwd(), "/.venv"))
utils <- reticulate::import("utils.challenges")
challenges <- utils$load_challenges()

challenges <- challenges[challenges$language == language, ]
challenges$tool <- "nametagger"
challenges$tokens <- NA

for (i in 1:nrow(challenges)) {
    annotations <- predict(model, challenges$text[i])
    challenges$tokens[i] <- list(annotations$term)
    challenges$iob[i] <- list(annotations$entity)

}

jsonlite::write_json(challenges, 'results/nametagger_challenges.json')
