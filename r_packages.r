pkg <- c("corpustools", 
         "caret", 
         "arrow", 
         "stringi", 
         "udpipe", 
         "nametagger", 
         "dataverse", 
         "showtext", 
         "patchwork", 
         "tidyverse", 
         "RColorBrewer",
         "reticulate")

for (p in pkg) {
    install.packages(p)
}

