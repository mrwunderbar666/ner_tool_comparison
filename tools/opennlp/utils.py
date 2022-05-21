from subprocess import Popen, PIPE

apache2conll = {'B-location': 'B-LOC', 'I-location': 'I-LOC',
                'B-person': 'B-PER', 'I-person': 'I-PER',
                'B-organization': 'B-ORG', 'I-organization': 'I-ORG',
                'B-misc': 'B-MISC', 'I-misc': 'I-MISC', }

def annotate(sentences, opennlp_bin, model='en-ner-person'):
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