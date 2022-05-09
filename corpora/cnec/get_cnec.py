import requests
import zipfile
from pathlib import Path
import shutil
from tqdm import tqdm
from nltk.corpus.reader import XMLCorpusReader

p = Path.cwd() / 'corpora' / 'cnec'
tmp = p / 'tmp'

def downloader(response, destination):
    total_size = int(response.headers.get('content-length', 0))
    with open(destination, 'wb') as f:
        with tqdm(total=total_size) as pbar:
            for chunk in response.iter_content(chunk_size=1024):
                b = f.write(chunk)
                pbar.update(b)

if not tmp.exists():
    tmp.mkdir()

repo = "https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11858/00-097C-0000-0023-1B22-8/Czech_Named_Entity_Corpus_2.0.zip"

print(f'Downloading Czech Named Entity Corpus 2.0 from: {repo}...')
r = requests.get(repo, stream=True)
downloader(r, tmp / 'Czech_Named_Entity_Corpus_2.0.zip')
print('Success!')

z = zipfile.ZipFile(tmp / 'Czech_Named_Entity_Corpus_2.0.zip', mode='r')
z.extractall(path=tmp)

cnec = XMLCorpusReader(str(tmp / 'cnec2.0' / 'data' / 'treex'), ['named_ent_dtest.treex'])


def heap(l):
    stack = []
    for i, char in enumerate(l):
        if char == '<':
            stack.append(i)
        elif char == '>':
            prev_pos = stack.pop()
            yield (len(stack), prev_pos, i)
        

def parse(li):
    tags = [x for x in heap(li)]

with open(str(tmp / 'cnec2.0' / 'data' / 'xml' / 'named_ent_dtest.xml')) as f:
    lines = f.readlines()

newlines = []
for l in lines[1:-1]:
    newlines.append('<sentence>' + l.replace('\n', '') + '</sentence>\n')

with open(str(tmp / 'named_ent_dtest.xml'), 'w') as f:
    f.write('<doc>\n')
    f.writelines(newlines)
    f.write('</doc>')

from bs4 import BeautifulSoup


def recurse_children(node, tok_dict, level=0):
    for tag in node.find_all('children', recursive=False):
        a_ref = tag.find_all('a.rf', recursive=False)
        lm_top = tag.find_all('LM', recursive=False)
        if len(a_ref) >  0:
            for a_ref in tag.find_all('a.rf', recursive=False):
                if len(a_ref.find_all('LM')) > 0:
                    for lm in a_ref.find_all('LM'):
                        tok_dict[lm.get_text()][f'NER_{level}'] = a_ref.parent.ne_type.get_text()
                else:
                    tok_dict[lm.get_text()][f'NER_{level}'] = a_ref.parent.ne_type.get_text()
        elif len(lm_top) > 0:
            for lm in lm_top:
                if len(lm.find_all('LM')) > 0:
                    for lm_sub in lm.find_all('LM'):
                        tok_dict[lm_sub.get_text()][f'NER_{level}'] = lm.ne_type.get_text()
                else:
                    tok_dict[lm.find('a.rf').get_text()][f'NER_{level}'] = lm.ne_type.get_text()
        if len(tag.find_all('children', recursive=False)) > 0:
            tok_dict = recurse_children(tag, tok_dict, level=level + 1)
    return tok_dict 


with open(str(tmp / 'cnec2.0' / 'data' / 'treex' / 'named_ent_dtest.treex')) as f:
    soup = BeautifulSoup(f, 'xml')

sentences = soup.bundles.find_all("LM", recursive=False)


tokenized = []
tokens = {}

for sentence in sentences:
    new_tokens = {lm['id']: 
                {'sentence_id': sentence['id'],
                'token_id': lm['id'],
                'token': lm.form.get_text(),
                'position': lm.ord.get_text(),
                'NER_0': 'None'} for lm in sentence.a_tree.find_all('LM', id=True)} 
    tokens.update(new_tokens)


for a in soup.find_all('a.rf'):
    if len(a.findChildren()) == 0:
        tokens[a.get_text()]['NER_0'] = a.parent.ne_type.get_text()
    else:
        for sub in a.findChildren():
            tokens[sub.get_text()]['NER_0'] = a.parent.ne_type.get_text()
