
### Helper Functions

# Recode CoNLL Corpus Named Entity Columns
recode_iob <- function(df, colname='CoNLL_IOB2') {
  
  # Recode Code Columns
  # Column for any named entity
  df$any_NE <- as.numeric(df[[colname]] != 'O') 
  
  # Columns without I-* or B-*
  df$NE <- gsub("^[BI]-", "", df[[colname]])
  df$NE <- as.factor(df$NE)
  return(df)
}

# Recode Dictionary Results
recode_results <- function(df) {
  
  codings <- as.data.frame(df)
  
  if ("type" %in% colnames(codings)) {
  
    codings$JRC_any_NE <- as.numeric(!is.na(codings$type))
    codings[is.na(codings$type), ]$type <- ""
    codings$JRC_NE <- "O"
    
    filt_persons <- codings$type == 'P'
    codings[filt_persons, 'JRC_NE'] <- "PER"
    
    filt_orgs <- codings$type == 'O'
    codings[filt_orgs, 'JRC_NE'] <- "ORG"
    
  } else {
    codings$JRC_any_NE <- 0
    codings$JRC_NE <- "O"
  }
  codings$JRC_NE <- factor(codings$JRC_NE, levels=levels(codings$NE))
  
  return(codings)  
}

# Extract Relevant Information from caret::confusionMatrix
cm2df <- function(confusion, corpus, lang) {
  if (is.null(confusion$positive)) {
    return(data.frame(
      corpus = corpus, 
      lang = lang, 
      task = gsub("Class: ", "", names(confusion$byClass[, 'Precision'])), 
      precision = unname(confusion$byClass[, 'Precision']),
      recall = unname(confusion$byClass[, 'Recall']),
      specificity = unname(confusion$byClass[, 'Specificity']),
      f1 = unname(confusion$byClass[, 'F1'])
    ))
    
  }
    else {
      return(data.frame(
        corpus = corpus, 
        lang = lang, 
        task = confusion$positive, 
        precision = unname(confusion$byClass['Precision']),
        recall = unname(confusion$byClass['Recall']),
        specificity = unname(confusion$byClass['Specificity']),
        f1 = unname(confusion$byClass['F1'])
      ))
  }
}


