import typing
from pathlib import Path
from subprocess import Popen, PIPE

def annotate(sentences: typing.Union[str, list], 
             opennlp_bin: typing.Union[Path, str], 
             model: typing.Union[str, Path]='en-ner-person') -> list:
    if isinstance(sentences, list):
        sentences = "\n".join(sentences)

    p = Popen([opennlp_bin, "TokenNameFinder", model],
                    shell=False, stdin=PIPE, stdout=PIPE, stderr=PIPE, universal_newlines=True)

    (stdout, stderr) = p.communicate(sentences)

    annotated = []

    for tagged in stdout.split('\n'):
        if tagged.strip == '': continue
        ner_tag = 'O'
        tags = []
        first = False
        for token in tagged.split(' '):
            if token.strip() == '': continue
            if token.startswith('<START:'):
                ner_tag = token.replace('<START:', '').replace('>', '')
                first = True
                continue
            elif token == '<END>':
                ner_tag = 'O'
                continue
            if first:
                tags.append('B-' + ner_tag)
                first = False
            elif ner_tag != 'O':
                tags.append('I-' + ner_tag)
            else:
                tags.append(ner_tag)
        annotated.append(tags) if len(tags) > 0 else None

    return annotated