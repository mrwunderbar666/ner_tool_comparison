Repository: https://github.com/EuropeanaNewspapers/ner-corpora

git clone https://github.com/EuropeanaNewspapers/ner-corpora


# Create Patch File
diff -u corpora/europeana/tmp/ner-corpora-master/enp_DE.sbb.bio/enp_DE.sbb.bio corpora/europeana/enp_DE.sbb.bio.fixed > corpora/europeana/enp_DE.sbb.patch


# Issues

## enp_FR.bnf

- Low data quality: IOB2 tags only contain I-Tags (B-Tags are missing)

## enp_DE.sbb

- many problems, requires separate patch file to manually fix data structure

## enp_de.lft

- one tag is `B-BER`, but should be `B-PER`