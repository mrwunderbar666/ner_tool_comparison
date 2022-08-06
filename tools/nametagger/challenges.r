library(nametagger)
library(stringr)
library(jsonlite)

if (!dir.exists("tools/nametagger/tmp")) {
  dir.create("tools/nametagger/tmp")
}

if (!file.exists("tools/nametagger/tmp/english-conll-140408.ner")) { 
  model <- nametagger_download_model(language = "english-conll-140408", model_dir = "tools/nametagger/tmp")
} else {
  model <- nametagger_load_model("tools/nametagger/tmp/english-conll-140408.ner")
}

language <- "en"

challenges <- jsonlite::read_json("challenges.json")

for (i in 1:length(challenges)) {
  challenges[[i]]$tool <- "nametagger"

  if (challenges[[i]]$language == language) {
    annotations <- predict(model, challenges[[i]]$text)
    challenges[[i]]$tokens <- list(annotations$term)
    challenges[[i]]$iob <- list(annotations$entity)

  }
}

jsonlite::write_json(challenges, 'results/nametagger_challenges.json')
