import sys
from pathlib import Path
import pandas as pd
import requests
from timeit import default_timer as timer
from datetime import timedelta
from time import sleep

sys.path.insert(0, str(Path.cwd()))
from tools.corenlp.utils import launch_server, annotate_sentence
from utils.mappings import corenlp2conll

languages = {'ar': 'arabic',
             'zh': 'chinese', 
             'en': 'english', 
             'fr': 'french', 
             'de': 'german', 
             'hu': 'hungarian',
             'it': 'italian',
             'es': 'spanish',
             }

examples = {'ar': ["فِييَنَّا هي عاصمة النمسا وأكبر مدنها من حيث عدد السكان. وسميت بهذا تطويرا عن اسمها اللاتيني القديم (فيندوبونا) ومعناه الهواء الجميل أو النسيم العليل. يبلغ عدد سكان فيينا حوالي 1.7 مليون نسمة، وهي عاشر أكبر مدينة من ناحية عدد السكان في الاتحاد الأوروبي."],
            'en': ['President Obama went to Beijing on Sunday.',
                   "One clear sign of Japan's nervousness came this week, when a spokesman for Japan's Foreign Ministry devoted nearly all of a regular, half-hour briefing for foreign journalists to the subject of recent Japanese investments in the U.S.",
                   """President-elect Kim Dae Jung today blamed much of Asia's devastating financial crisis on governments that "lie" to their people and "authoritarian" leaders who place economic growth ahead of democratic freedoms."""],
            'zh': ["习主席周日前往北京。",
                   "該城是德語圈中的第二大城市，僅次於柏林",
                   "2001年維也納市中心古城區被指定為聯合國世界遺產，2017年7月它被移至瀕危世界遺產的名錄中。"
                   ],
            'fr': ["Charles de Gaulle s'est rendu dimanche à Pékin.",
                   "La ville est située dans l'est du pays et traversée par le Danube.",
                   "Vienne est un important centre politique international, notamment en raison de la neutralité autrichienne, puisqu'y siègent l'OSCE, l'OPEP et diverses agences de l'ONU, comme l'Agence internationale de l'énergie atomique, l'Office des Nations unies contre la drogue et le crime ou l'ONUDI."],
            'de': ["Helmut Kohl reiste am Sonntag nach Peking",
                   "Mit rund 2 Millionen Einwohnern (2023) – etwas mehr als einem Fünftel der österreichischen Gesamtbevölkerung – ist das an der Donau gelegene Wien die bevölkerungsreichste Großstadt und Primatstadt Österreichs sowie die zweitgrößte Stadt des deutschen Sprachraums und die fünftgrößte Stadt der Europäischen Union.",
                   "So ist Wien heute als internationaler Kongress- und Tagungsort Sitz von über 40 internationalen Organisationen, darunter das Erdölkartell OPEC, die Internationale Atomenergiebehörde IAEO und die OSZE, und zählt damit zu den Weltstädten."],
            'hu': ["Orbán Viktor vasárnap Pekingbe ment.",
                   "A város a középkorban sem vesztett jelentőségéből, majd később az Osztrák–Magyar Monarchia fővárosa lett Budapest mellett, egészen a Monarchia első világháborút követő felbomlásáig.",
                   "A város közel fekszik a cseh, a szlovák és a magyar határokhoz, és három nagyobb városhoz, Pozsonyhoz, Brnóhoz és Győrhöz."],
            'it': ["Matteo Salvini domenica è andato a Pechino.",
                   "Sede di importanti organizzazioni internazionali tra le quali: l'Organizzazione dei Paesi esportatori di petrolio (OPEC), l'Agenzia internazionale per l'energia atomica (AIEA) e l'Organizzazione delle Nazioni Unite (ONU) con il centro storico della città che è stato dichiarato patrimonio dell'umanità dall'UNESCO, è anche un centro industriale con, principalmente, industrie elettroniche, tessili, agroalimentari, siderurgiche, chimiche, meccaniche di precisione.",
                   "Vienna è la capitale dell'Austria e allo stesso tempo uno dei suoi nove Stati federati, completamente circondato dalla Bassa Austria, è il sesto comune per abitanti dell'Unione europea."],
            'es': ["Ronaldo viajó a Beijing el domingo.",
                   "Viena es una ciudad austriaca situada a orillas del Danubio, en el valle de los Bosques de Viena, al pie de las primeras estribaciones de los Alpes.",
                   "En la década de 1970, el canciller austriaco Bruno Kreisky inauguró el Centro Internacional de Viena, una nueva área de la ciudad creada para albergar instituciones internacionales."]
                   }

p = Path.cwd() / 'tools' / 'corenlp'

corenlp_folder = list(p.glob('stanford-corenlp-*'))[0]

for lang, language in languages.items():
    print('Testing language:', language)
    
    corenlp_server = launch_server(corenlp_folder, language=language)
    corenlp_ready = False
    server_address = 'http://localhost:9000/'

    while not corenlp_ready:
        try:
            r = requests.get(server_address + 'ready')
            if r.status_code == 200:
                corenlp_ready = True
        except:
            print('waiting for server...')
        finally:
            sleep(0.5)

    # Send a test sentence to provoke CoreNLP to load all files
    params = {'properties': '{"annotators":"ner","outputFormat":"json","tokenize.language": "Whitespace"}'}
    sentence = 'This is a testing sentence.'
    r = requests.post(server_address, params=params, data=sentence)

    assert r.status_code == 200, 'CoreNLP Server not responding!'

    params = {'properties': '{"annotators":"ner","outputFormat":"json"}'}

    for sentence in examples[lang]:
        print('***')
        print(sentence)
        tokens, iob = annotate_sentence(sentence, params=params)
        for t, i in zip(tokens, iob):
            print(t, '\t', i)


    print('-'*80)

print('Done!')