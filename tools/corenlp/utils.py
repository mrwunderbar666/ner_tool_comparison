
import requests
import subprocess
from tqdm import tqdm
import psutil

from pandas import DataFrame

def launch_server(corenlp_folder, language='english'):
    corenlp_server = [p for p in psutil.process_iter() if 'edu.stanford.nlp.pipeline.StanfordCoreNLPServer' in p.cmdline()]
    if len(corenlp_server) > 0:
        print('Found other instance running. Terminating old processes...')
        [p.kill() for p in corenlp_server]

    args = ['java', '-mx4g', '-cp', '*', 'edu.stanford.nlp.pipeline.StanfordCoreNLPServer', '-port', '9000', '-timeout', '15000', '-quiet']
    if language != 'english':
        args += ['-serverProperties', f'StanfordCoreNLP-{language}.properties']

    # java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -serverProperties StanfordCoreNLP-chinese.properties -port 9000 -timeout 15000
    print('Launching CoreNLP server...')
    corenlp_server = subprocess.Popen(args=args, 
                                        cwd=corenlp_folder, 
                                        stdout=subprocess.DEVNULL,
                                        stderr=subprocess.STDOUT)
    
    return corenlp_server

def annotate(df: DataFrame, 
                col_sentence_id = "sentence_id", 
                server_address = 'http://localhost:9000/',
                params = {'properties': '{"annotators":"ner","outputFormat":"json","tokenize.language": "Whitespace"}'}
                ):
    assert all([i in df.columns for i in ["token", "token_id", col_sentence_id]]), 'DataFrame has wrong format!'
    
    df['corenlp_ner'] = 'O'

    with tqdm(df[col_sentence_id].unique(), unit="sentence") as pbar:
        for sentence_id in df[col_sentence_id].unique():
            filt_sentence = (df[col_sentence_id] == sentence_id)
            sentence = " ".join(df.loc[filt_sentence, 'token'].tolist())
            r = requests.post(server_address, params=params, data=sentence.encode('utf-8'))
            j = r.json()
            for entity in j['sentences'][0]['entitymentions']:
                ner_type = entity['ner']
                first = True
                for i in range(entity['tokenBegin'], entity['tokenEnd']):
                    if j['sentences'][0]['tokens'][i]['ner'] == 'O':
                        # somehow the JSON output of CoreNLP is inconsistent
                        # the 'entitymentions' also includes personal pronouns
                        # but the 'tokens' do not call these 'ner' ¯\_(ツ)_/¯
                        continue
                    if first:
                        ner = 'B-' + ner_type
                    else:
                        ner = 'I-' + ner_type
                    filt_token = (df.token_id == i+1)
                    df.loc[filt_sentence & filt_token, 'corenlp_ner'] = ner
                    first = False
            pbar.update(1)


def annotate_sentence(sentence: str, 
                server_address = 'http://localhost:9000/',
                params = {'properties': '{"annotators":"ner","outputFormat":"json","tokenize.language": "Whitespace"}'}
                ):
    
    r = requests.post(server_address, params=params, data=sentence.encode('utf-8'))
    j = r.json()
    iob = len(j['sentences'][0]['tokens']) * ['O']
    tokenized = [tok['originalText'] for tok in j['sentences'][0]['tokens']]
    for entity in j['sentences'][0]['entitymentions']:
        ner_type = entity['ner']
        first = True
        for i in range(entity['tokenBegin'], entity['tokenEnd']):
            if j['sentences'][0]['tokens'][i]['ner'] == 'O':
                # somehow the JSON output of CoreNLP is inconsistent
                # the 'entitymentions' also includes personal pronouns
                # but the 'tokens' do not call these 'ner' ¯\_(ツ)_/¯
                continue
            if first:
                ner = 'B-' + ner_type
            else:
                ner = 'I-' + ner_type
            iob[i] = ner
            first = False
    return tokenized, iob