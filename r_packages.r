pkg <- c("corpustools", "caret", "arrow", "stringi")

for (p in pkg) {
  if (nchar(find.package(p)) == 0) {
    install.packages(p)
  }
}

