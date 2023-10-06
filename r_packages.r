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
         "reticulate",
         "kableExtra",
         "viridis")

for (p in pkg) {
    install.packages(p)
}

