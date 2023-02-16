# CoNLL-2002 (Spanish & Dutch)

- Website: https://www.clips.uantwerpen.be/conll2002/ner/
- Citation: Introduction to the CoNLL-2002 Shared Task: Language-Independent Named Entity Recognition (Tjong Kim Sang, 2002). https://aclanthology.org/W02-2024/

## Get the Data

Run

```bash
python corpora/conll/get_conll2002.py
```

# CoNLL-2003 (English & German)

- Website: https://www.clips.uantwerpen.be/conll2003/ner/
- Citation: Introduction to the CoNLL-2003 Shared Task: Language-Independent Named Entity Recognition (Tjong Kim Sang & De Meulder, 2003), https://aclanthology.org/W03-0419/




## Getting the English dataset

Two options
- CoNLL++ (English only): https://github.com/ZihanWangKi/CrossWeigh
- CoNLL 2003 English: https://huggingface.co/datasets/conll2003

Getting the original dataset via huggingface:

```bash
python corpora/conll/conll2003_en_huggingface.py
```

## Getting the German dataset

Two Options:
- Get the [official German resource](https://catalog.ldc.upenn.edu/LDC94T5) and run the steps described below.
- Get the already processed dataset from huggingface, which is a legal grey-zone. CoNLL 2003 German: https://huggingface.co/datasets/Davlan/conll2003_de_noMISC/tree/main

**Disclaimer:** This probably won't work in Windows. If you're doing this in Windows, you can try using a Linux Subsystem.

1. Get German Base Data from: https://catalog.ldc.upenn.edu/LDC94T5 (costs USD 75 for non-members)
2. Download the NER build tools from the original website
    - https://www.clips.uantwerpen.be/conll2003/ner.tgz
    - in case website is down, web.archive.org still has the files
3. Extract all archives in separate folders
4. Open the file `ner/bin/make.deu` with a text editor and make several changes
    - change the location of the "cd rom" to the actual path were you extracted the LDC94T5 package. E.g., change `CORPUS="/mnt/cdrom/data/eci1/ger03/ger03b05.eci"` to `CORPUS="/home/user/eci_multilang_txt/data/eci1/ger03/ger03b05.eci"`
    - change the line stating `DIR="../etc"` to `DIR="../etc.2006"`. This makes sure the script uses the corrected annotations, since the original annotations contain a few errors.
    - comment out the line starting with `grep -v "^ *$" |\` by just adding a `#` at the beginning. Right after this line add the following: `sed -e /^$/d |\`. For some reason `grep` gave me unexpected results and skipped over lines that started with a quotation mark `"`. Using `sed` instead did the trick. 
5. Open a terminal in latin-1 (iso-8859-1) and change into the main directory of the extracted `ner.tgz`
6. run the main script with `bin/make.deu`
7. You should now have three new files in your current directory: `deu.train`, `deu.testa`, `deu.testb`
8. Copy the files into subdirectory for this repository: `corpora/conll`
9. Finally run `python corpora/conll/convert_conll2003_de.py`

In case more details are needed, check the instructions in `ner.tgz` (`000README`)

Similar problem on github: https://github.com/flairNLP/flair/issues/1102