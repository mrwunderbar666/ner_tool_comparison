library(corpustools)
library(readr)
library(arrow)
library(stringi)
library(caret)
library(jsonlite)
library(dplyr)

challenges <- jsonlite::fromJSON("challenges.json")

challenges$doc_id <- 1:nrow(challenges)
challenges$doc_id <- stringi::stri_pad(challenges$doc_id, 2, pad="0")


# Load ICEWS Names Dictionary
icews_actors <- read_rds('tools/icews/icews_actors.rds')


tc <- corpustools::create_tcorpus(challenges)
tc$code_dictionary(
  icews_actors,
  case_sensitive = T,
  token_col = 'token',
  string_col = 'keyword',
  sep = ' ',
  use_wildcards = F
)

tc$tokens$token <- as.character(tc$tokens$token) 

results <- tc$tokens %>% group_by(doc_id) %>% summarise(tokens = list(token), code_id = list(code_id)) %>% right_join(challenges, by="doc_id")

# Save results

if (!dir.exists('results')) {
  dir.create('results')
}

jsonlite::write_json(results, "results/icews_challenges.json")
