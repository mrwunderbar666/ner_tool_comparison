from bs4 import BeautifulSoup
import copy
from pathlib import Path
import typing as t

def parse_xml(path: Path) -> t.List[dict]:
    with open(path) as f:
        soup = BeautifulSoup(f.read(), 'xml')

    document_id = path.parent.name + '_' + path.name.replace('.xml', '')

    article = []
    for i, sentence in enumerate(soup.find_all('sentence')):
        s = [token for token in sentence.find_all(wd=True)]
        for token in s:
            token['sentence_number'] = i
            token['document_id'] = document_id
            if "_" in token['wd']:
                bio = 'B'
                for subtoken in token['wd'].split('_'):
                    newtoken = copy.copy(token)
                    newtoken['wd'] = subtoken
                    newtoken['bio'] = bio
                    article.append(newtoken.attrs)
                    bio = 'I'
            else:
                article.append(token.attrs)
    
    return article

s = parse_xml(Path("corpora/ancora/ancora-es-2.0.0_2/ancora-2.0/3LB-CAST/t5-4.tbf.xml"))
print(s)

