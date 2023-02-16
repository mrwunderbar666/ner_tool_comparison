# WikiANN

- Citation: Cross-lingual Name Tagging and Linking for 282 Languages (Pan et al., ACL 2017). https://aclanthology.org/P17-1178/
- Repository: https://elisa-ie.github.io/wikiann/
- Dataset: https://drive.google.com/drive/folders/1Q-xdT99SeaCghihGa7nRkcXGwRGUIsKN
- Huggingface Dataset: https://huggingface.co/datasets/wikiann
- Automatic Dataset Creation: https://github.com/panx27/wikiann

# Get the Data

1. Go to his Google Drive Folder: https://drive.google.com/drive/folders/1aSpKIvRJUiOO3oPSY8A-5R2vI4eg6X96
2. Right-click on the folder `name_tagging` and click download
3. Copy the .zip file into the wikiann folder
4. run the script `get_wikiann.py`
  - The script automatically generates training, test, and validation splits for each language (with a random state for reproducability)
  - It splits into train (70%), test (15%), validate (15%), but with a  maximum number of test / validation sample of 25,000 sentences each
