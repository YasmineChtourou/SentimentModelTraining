from nltk.stem import WordNetLemmatizer
import gensim
from gensim import parsing
from gensim.parsing.preprocessing import split_alphanum
from spellchecker import SpellChecker
import re


def normaliser_word(word):
    slangs_dict = {
    'awsm': 'awesome',
    "aamof": "as a matter of fact",
    "abt": "about",
    "abt2": "about to",
    "ac": "air conditioning",
    "ace": "solo winner",
    "ack": "acknowledged",
    "admin": "administrator",
    "thr": "there",
    "frm": "from",
    "aggro": "aggression",
    "agl": "angel",
    "dob": "date of birth",
    "ai": "artificial intelligence",
    "aiic": "as if i care",
    "aka": "also known as",
    "alap": "as long as possible",
    "alol": "actually laughing out loud",
    "ama": "ask me anything",
    "amap": "as much as possible",
    "amazn": "amazing",
    "ammo": "ammunition",
    "ams": "ask me something",
    "anon": "anonymous",
    "asap": "as soon as possible",
    "asat": "as simple as that",
    "awks": "awkward",
    "awl": "always with love",
    "ayk": "as you know",
    "azm": "awesome",
    "b": "be",
    "b&w": "black and white",
    "b-day": "birthday",
    "bday": "birthday",
    "bcoz": "because",
    "bcos": "because",
    "bcz": "because",
    "bf": "boyfriend",
    "btw": "between",
    "b4": "before",
    "bai": "bye",
    "bb": "bye bye",
    "bc": "abuse",
    "mc": "abuse",
    "bcc": "blind carbon copy",
    "bff": "best friends forever",
    "biz": "business",
    "bk": "back",
    "bo": "back off",
    "bro": "brother",
    "btwn": "between",
    "c": "see",
    "char": "character",
    "combo": "combination",
    "cu": "see you",
    "cu2": "see you too",
    "cu2mr": "see you tomorrow",
    "cya": "see ya",
    "cyal8r": "see you later",
    "cyb": "call you back",
    "cye": "check your e-mail",
    "cyf": "check your facebook",
    "cyfb": "check your facebook",
    "cyl": "catch ya later",
    "cym": "check your myspace",
    "cyo": "see you online",
    "d8": "date",
    "da": "the",
    "dece": "decent",
    "ded": "dead",
    "dept": "department",
    "dis": "this",
    "ditto": "same",
    "diva": "rude woman",
    "dk": "don't know",
    "dlm": "don't leave me",
    "dm": "direct message",
    "dnd": "do not disturb",
    "dno": "don't know",
    "dnt": "don't",
    "e1": "everyone",
    "eg": "for example",
    "emc2": "genius",
    "emo": "emotional",
    "enuf": "enough",
    "eod": "end of discussion",
    "eof": "end of file",
    "eom": "end of message",
    "eta": "estimated time of arrival",
    "every1": "everyone",
    "evs": "whatever",
    "exp": "experience",
    "f": "female",
    "f2f": "face to face",
    "f2p": "free to play",
    "f2t": "free to talk",
    "f9": "fine",
    "fab": "fabulous",
    "fail": "failure",
    "faq": "frequently asked questions",
    "fav": "favorite",
    "fave": "favorite",
    "favs": "favorites",
    "fb": "facebook",
    "fbc": "facebook chat",
    "fbf": "facebook friend",
    "fml": "family",
    "fn": "fine",
    "fo": "freaking out",
    "fri": "friday",
    "frnd": "friend",
    "fu": "fuck you",
    "fugly": "fucking ugly",
    "gf": "girlfriend",
    "g1": "good one",
    "g2b": "going to bed",
    "g2cu": "good to see you",
    "g2g": "good to go",
    "g4i": "go for it",
    "g4n": "good for nothing",
    "g4u": "good for you",
    "g9": "goodnight",
    "ga": "go ahead",
    "ge": "good evening",
    "gl": "good luck",
    "gm": "good morning",
    "gn": "goodnight",
    "gonna": "going to",
    "goon": "idiot",
    "gorge": "gorgeous",
    "gr8": "great",
    "grats": "congratulations",
    "gratz": "congratulations",
    "grl": "girl",
    "gt2t": "got time to talk",
    "gtg": "good to go",
    "gud": "good",
    "gv": "give",
    "gvn": "given",
    "gw": "good work",
    "h/o": "hold on",
    "h/p": "hold please",
    "h/t": "hat tip",
    "h/u": "hook up",
    "h2cus": "hope to see you soon",
    "h4u": "hot for you",
    "h4x0r": "hacker",
    "h4x0rz": "hackers",
    "h8": "hate",
    "h8r": "hater",
    "h8t": "hate",
    "ha": "hello again",
    "haha": "laughing",
    "hai": "hi",
    "hak": "hugs and kisses",
    "han": "how about now?",
    "hav": "have",
    "hax": "hacks",
    "haxor": "hacker",
    "hay": "how are you",
    "hb2u": "happy birthday to you",
    "hbbd": "happy belated birthday",
    "hbd": "happy birthday",
    "hc": "how cool",
    "hcit": "how cool is that",
    "hehe": "laughing",
    "hf": "have fun",
    "hi5": "high five",
    "hig": "how's it going?",
    "hih": "hope it helps",
    "ho": "hold on",
    "hoc": "house of cards",
    "hof": "hall of fame",
    "holla": "holler",
    "hom": "hit or miss",
    "hood": "neighborhood",
    "hoops": "basketball",
    "hottie": "attractive person",
    "hr": "human resources",
    "hru": "how are you",
    "hry": "hurry",
    "hubby": "husband",
    "hwk": "homework",
    "hwp": "height weight proportionate",
    "hwu": "hey, what's up?",
    "hxc": "hardcore",
    "h^": "hook up",
    "i8": "i ate",
    "i8u": "i hate you",
    "ia": "i agree",
    "iab": "in a bit",
    "iac": "in any case",
    "iad": "it all depends",
    "iae": "in any event",
    "iag": "it's all good",
    "iagw": "in a good way",
    "iail": "i am in love",
    "iam": "in a minute",
    "ic": "i see",
    "id10t": "idiot",
    "idc": "i don't care",
    "idd": "indeed",
    "idi": "i doubt it",
    "idk": "i don't know",
    "idky": "i don't know why",
    "idmb": "i'll do my best",
    "idn": "i don't know",
    "idnk": "i do not know",
    "idr": "i don't remember",
    "idt": "i don't think",
    "idts": "i don't think so",
    "idtt": "i'll drink to that",
    "idu": "i don't understand",
    "ie": "that is",
    "ig2p": "i got to pee",
    "iggy": "ignored",
    "ight": "alright",
    "igi": "i get it",
    "ign": "in-game name",
    "igtp": "i get the point",
    "ih8u": "i hate you",
    "ihu": "i hate you",
    "ihy": "i hate you",
    "ii": "i'm impressed",
    "iiok": "if i only knew",
    "iir": "if i remember",
    "iirc": "if i remember correctly",
    "iit": "i'm impressed too",
    "iiuc": "if i understand correctly",
    "ik": "i know",
    "ikhyf": "i know how you feel",
    "ikr": "i know, right?",
    "ikwum": "i know what you mean",
    "ikwym": "i know what you mean",
    "ikyd": "i know you did",
    "ilu": "i like you",
    "ilu2": "i love you too",
    "ilub": "i love you baby",
    "ilyk": "i'll let you know",
    "ilyl": "i love you lots",
    "ilysm": "i love you so much",
    "ima": "i'm",
    "imma": "i'm gonna",
    "imo": "in my opinion",
    "imy": "i miss you",
    "inb4": "in before",
    "inc": "incoming",
    "indie": "independent",
    "info": "information",
    "init": "initialize",
    "ipo": "initial public offering",
    "ir": "in room",
    "ir8": "irate",
    "irdk": "i really don't know",
    "irl": "in real life",
    "iyo": "in your opinion",
    "iyq": "i like you",
    "j/k": "just kidding",
    "j/p": "just playing",
    "j/w": "just wondering",
    "j2lyk": "just to let you know",
    "j4f": "just for fun",
    "j4g": "just for grins",
    "jas": "just a second",
    "jb/c": "just because",
    "joshing": "joking",
    "k": "ok",
    "k3u": "i love you",
    "kappa": "sarcasm",
    "kek": "korean laugh",
    "keke": "korean laugh",
    "kewl": "cool",
    "kewt": "cute",
    "kfc": "kentucky fried chicken",
    "kgo": "ok, go",
    "kik": "laughing out loud",
    "kinda": "kind of",
    "kk": "ok",
    "kl": "kool",
    "km": "kiss me",
    "kma": "kiss my ass",
    "knp": "ok, no problem",
    "kw": "know",
    "kwl": "cool",
    "l2m": "listening to music",
    "l2p": "learn to play",
    "l33t": "leet",
    "l8": "late",
    "l8er": "later",
    "l8r": "later",
    "la": "laughing a lot",
    "laf": "laugh",
    "laffing": "laughing",
    "lafs": "love at first sight",
    "lam": "leave a message",
    "lamer": "lame person",
    "legit": "legitimate",
    "lemeno": "let me know",
    "lil": "little",
    "lk": "like",
    "llol": "literally laughing out loud",
    "lmho": "laughing my head off",
    "loi": "laughing on the inside",
    "lola": "love often, laugh a lot",
    "lolol": "lots of laugh out louds",
    "lolz": "laugh out louds",
    "ltr": "later",
    "lulz": "lol",
    "luv": "love",
    "luzr": "loser",
    "lv": "love",
    "ly": "love ya",
    "lya": "love you always",
    "lyk": "let you know",
    "lyn": "lying",
    "lysm": "love you so much",
    "m": "male",
    "mcd": "mcdonald's",
    "mcds": "mcdonald's",
    "md@u": "mad at you",
    "me2": "me too",
    "meh": "whatever",
    "mf": "mother fucker",
    "mfb": "mother fucking bitch",
    "mgmt": "management",
    "mid": "middle",
    "mil": "mother-in-law",
    "min": "minute",
    "mins": "minutes",
    "mk": "okay",
    "mkay": "ok",
    "mmk": "ok",
    "mms": "multimedia messaging service",
    "mng": "manage",
    "mngr": "manager",
    "mod": "modification",
    "mofo": "mother fucking",
    "mojo": "attractive talent",
    "moss": "chill",
    "ms": "miss",
    "msg": "message",
    "mtg": "meeting",
    "mth": "month",
    "mu": "miss you",
    "mu@": "meet you at",
    "muah": "kiss",
    "mula": "money",
    "mwa": "kiss",
    "mwah": "kiss",
    "n/m": "nevermind",
    "n/m/h": "nothing much here",
    "n/r": "no reserve",
    "n00b": "newbie",
    "n1": "nice one",
    "n1c": "no one cares",
    "n2m": "not too much",
    "n2mh": "not too much here",
    "n2w": "not to worry",
    "n64": "nintendo 64",
    "n8kd": "naked",
    "nac": "not a chance",
    "nah": "no",
    "nal": "nationality",
    "narc": "tattle tale",
    "nark": "informant",
    "naw": "no",
    "nb": "not bad",
    "nbd": "no big deal",
    "nbjf": "no brag, just fact",
    "nd": "and",
    "ne": "any",
    "ne1": "anyone",
    "ne1er": "anyone here",
    "neh": "no",
    "nemore": "anymore",
    "neva": "never",
    "neway": "anyway",
    "newaze": "anyways",
    "newb": "newbie",
    "nite": "night",
    "nn2r": "no need to reply",
    "nnito": "not necessarily in that order",
    "nnto": "no need to open",
    "nntr": "no need to reply",
    "no1": "no one",
    "noob": "newbie",
    "nooblet": "young newbie",
    "nooblord": "ultimate newbie",
    "notch": "minecraft creator",
    "nottie": "unattractive person",
    "np": "no problem",
    "nub": "newbie",
    "nuff": "enough",
    "nufn": "nothing",
    "num": "tasty",
    "nvm": "nevermind",
    "nvr": "never",
    "nvrm": "nevermind",
    "nw": "no way",
    "nxt": "next",
    "o4u": "only for you",
    "obtw": "oh, by the way",
    "obv": "obviously",
    "obvi": "obviously",
    "oc": "of course",
    "ohemgee": "oh my gosh",
    "oic": "oh, i see",
    "oicn": "oh, i see now",
    "oiy": "hi",
    "omg": "oh my god",
    "onl": "online",
    "onoz": "oh no",
    "orly": "oh really",
    "otay": "okay",
    "otw": "on the way",
    "outta": "out of",
    "ovie": "overlord",
    "ownage": "completely owned",
    "p/d": "per day",
    "p/m": "per month",
    "p/y": "per year",
    "p911": "parent alert!",
    "p@h": "parents at home",
    "pc": "personal computer",
    "pda": "public display of affection",
    "pic": "picture",
    "pj": "poor joke",
    "pl8": "plate",
    "pld": "played",
    "pls": "please",
    "plz": "please",
    "plzrd": "please read",
    "pov": "point of view",
    "ppl": "people",
    "ppp": "peace",
    "prof": "professor",
    "prolly": "probably",
    "promo": "promotion",
    "props": "recognition",
    "prot": "protection",
    "prvt": "private",
    "ps": "postscript",
    "ps2": "playstation 2",
    "ps3": "playstation 3",
    "psa": "public service announcement",
    "psog": "pure stroke of genius",
    "psp": "playstation portable",
    "ptm": "please tell me",
    "pwd": "password",
    "psd": "password",
    "pswd": "password",
    "pwnd": "owned",
    "pwned": "owned",
    "pwnt": "owned",
    "q4u": "question for you",
    "qfe": "quoted for emphasis",
    "qft": "quoted for truth",
    "qq": "quick question",
    "qqn": "looking",
    "qrg": "quick reference guide",
    "qt": "cutie",
    "qtpi": "cutie pie",
    "r": "are",
    "r8": "rate",
    "rdy": "ready",
    "re": "replay",
    "rehi": "hi again",
    "rents": "parents",
    "rep": "reputation",
    "resq": "rescue",
    "rgd": "regard",
    "rgds": "regards",
    "ridic": "ridiculous",
    "rip": "rest in peace",
    "rl": "real life",
    "rlrt": "real life retweet",
    "rly": "really",
    "rm": "room",
    "rn": "run",
    "rnt": "aren't",
    "rof": "laughing",
    "rofl": "laughing",
    "roflmao": "laughing",
    "roflol": "laughing out loud",
    "rolf": "laughing",
    "ru": "are you",
    "ruc": "are you coming?",
    "rut": "are you there?",
    "rx": "prescription",
    "s/o": "sold out",
    "s/u": "shut up",
    "s/w": "software",
    "s2r": "send to receive",
    "s2s": "sorry to say",
    "s2u": "same to you",
    "samzd": "still amazed",
    "sd": "sweet dreams",
    "sec": "second",
    "sho": "sure",
    "sh^": "shut up",
    "siul8r": "see you later",
    "siv": "bad goaltender",
    "sk8": "skate",
    "sk8r": "skater",
    "sly": "still love you",
    "smf": "so much fun",
    "smooch": "kiss",
    "sorta": "sort of",
    "spec": "specialization",
    "spk": "speak",
    "spkr": "speaker",
    "srry": "sorry",
    "srs": "serious",
    "srsly": "seriously",
    "sry": "sorry",
    "stpd": "stupid",
    "str": "strength",
    "str8": "straight",
    "sup": "what's up",
    "syl": "see you later",
    "sync": "synchronize",
    "t2go": "time to go",
    "t2m": "talk to me",
    "t2u": "talk to you",
    "t2ul": "talk to you later",
    "t2ul8er": "talk to you later",
    "t2ul8r": "talk to you later",
    "t4lmk": "thanks for letting me know",
    "t4p": "thanks for posting",
    "t4t": "thanks for trade",
    "tc": "take care",
    "teh": "the",
    "teme": "tell me",
    "tg": "thank goodness",
    "thnq": "thank you",
    "tho": "though",
    "thru": "through",
    "tht": "that",
    "thx": "thanks",
    "tl": "tell",
    "tlk": "talk",
    "tlkin": "talking",
    "tlking": "talking",
    "tomoz": "tomorrow",
    "tq": "thank you",
    "tqvm": "thank you very much",
    "tru": "true",
    "ttl": "talk to you later",
    "ttly": "totally",
    "ttul": "talk to you later",
    "tty": "talk to you",
    "tu": "thank you",
    "tude": "attitude",
    "tx": "thanks",
    "txt": "text",
    "txtin": "texting",
    "ty": "thank you",
    "tyfa": "thank you for asking",
    "tyl": "thank you lord",
    "tym": "thank you much",
    "tyt": "take your time",
    "tyvm": "thank you very much",
    "u": "you",
    "u-ok": "you ok?",
    "u/l": "upload",
    "u2": "you too",
    "u2u": "up to you",
    "uok": "you ok?",
    "ur": "your",
    "ut": "you there?",
    "veggies": "vegetables",
    "vry": "very",
    "vs": "versus",
    "w/": "with",
    "w/b": "welcome back",
    "w/e": "whatever",
    "w/o": "without",
    "w2f": "way too funny",
    "w2g": "way to go",
    "w2k": "windows 2000",
    "w4u": "wait for you",
    "w8": "wait",
    "w84m": "wait for me",
    "w8am": "wait a minute",
    "w8ing": "waiting",
    "w8n": "waiting",
    "wa": "what",
    "waa": "crying",
    "wack": "strange",
    "wan2": "want to",
    "wannabe": "want to be",
    "wat": "what",
    "watev": "whatever",
    "watevs": "whatever",
    "wlcm": "welcome",
    "wha": "what",
    "whipped": "tired",
    "wht": "what",
    "wk": "week",
    "wknd": "weekend",
    "wtf": "what the fuck",
    "wtg": "way to go",
    "wup": "what's up?",
    "ya": "yes",
    "yeap": "yes",
    "yep": "yes",
    "yepperz": "yes",
    "yesh": "yes",
    "yo": "hi",
    "yr": "your",
    "yrs": "years",
    "yt": "you there?",
    "yt?": "you there?",
    "yup": "yes",
    "yupz": "ok",
    "zzz": "sleeping",
    }
    word = word.lower()
    word = word.split()
    for i in range(len(word)):
        word[i] = slangs_dict.get(word[i], word[i])
    word = " ".join(word)
    return word


def replace_word(word):
    switcher = {
        "couldn't": "could not",
        "couldn": "could not",
        "won't": "will not",
        "won": "will not",
        "mustn't": "must not",
        "mustn": "must not",
        "that'll": "that will",
        "shouldn't": "should not",
        "shouldn": "should not",
        "should've": "should have",
        "haven't": "have not",
        "haven": "have not",
        "hadn't": "have not",
        "hadn": "have not",
        "hasn't": "have not",
        "hasn": "have not",
        "didn't": "do not",
        "didn": "do not",
        "doesn't": "do not",
        "doesn": "do not",
        "don't": "do not",
        "don": "do not", 
        "isn't": "be not",
        "you'd":"you would",
        "you've":"you have",
        "you're":"you are",
        "you'll":"you will",
        "she's":"she is",
        "she'd":"she would",
        "she'll":"she will",
        "he's":"he is",
        "he'd":"he would",
        "he'll":"he will",
        "it's":"it is",
        "it'd":"it would",
        "it'll":"it will",
        "aren't":"are not",
        "aren":"are not",
        "weren't":"were not",
        "weren":"were not",
        "wouldn't":"would not",
        "wouldn":"would not",
        "needn't":"need not",
        "needn":"need not",
        "wasn't":"was not",
        "wasn":"was not",
        "mightn't":"might not",
        "mightn":"might not",
        "shan't":"shall not",
        "shan":"shall not", 
        "can't":"can not",
        "i'm":"i am",
        "i'd":"i would",
        "i'll":"i will",
        "i've":"i have",
        "we're":"we are",
        "we'd":"we would",
        "we'll":"we will",
        "we've":"we have",
        "they're":"they are",
        "they'd":"they would",
        "they'll":"they will",
        "they've":"they have",
        "let's":"let us",
        "how's":"how is",
        "here's":"here is",
        "what's":"what is",
        "there's":"there is",
        "0":"zero",
        "1":"one",
        "2":"two",
        "3":"three",
        "4":"four",
        "5":"five",
        "6":"six",
        "7":"oseven",
        "8":"eight",
        "9":"nine",
        "10":"ten",
    }
    word = word.lower()
    word = word.split()
    for i in range(len(word)):
        word[i] = switcher.get(word[i], word[i])
    word = " ".join(word)
    return word

def transform_text(text):
        text  = text.lower()
        text  = replace_word(text)
        text  = normaliser_word(text)
        text  = re.sub(r'[^\x00-\x7f]',r' ',text)
        stops={'at', 'only', 'your', 'yourself', 'a', 'i', 'during', 'off', 'myself', 'so', 'o', 'after', 'under', 
           'there', 'against', 'over', 'ourselves', 'they', 'me', 'its', 'then', 'above', 'theirs', 'this', 'into', 
           'from', 'very', 'on', 'yours', 'yourselves', 'herself', 'themselves', 'between', 'if', 'below', 'own', 
           'and', 'you', 'itself', 'him', 'while', 's', 'who', 'we', 'what', 'by', 'ma', 'further', 'such', 'until',
           'through', 'too', 'until', 'through', 't', 'too', 'where', 'up', 'my', 'm', 'out', 'down', 're', 'to', 
           'she', 'd', 'those', 'when', 'it', 'because', 'he', 'in', 'other','each', 'both', 'her', 'but', 'as', 'all', 
           'his', 'again', 'with', 'once', 'am', 'just', 'should', 'why', 'than', 'any', 'should', 'why', 'than',
           'more', 'most', 'that', 've', 'will', 'ours', 'our', 'll', 'the', 'y', 'which', 'whom', 'hers', 'an', 'here',
           'how', 'before', 'about', 'for', 'them', 'these', 'their', 'for', 'them', 'these', 'their', 'or', 'must', 
           'shall', 'would', 'could' , 'need', 'might'}
        filtered_words = [word for word in text.split() if word not in stops]
        filtered_words = gensim.corpora.textcorpus.remove_short(filtered_words, minsize=3)
        text = " ".join(filtered_words)
        text = gensim.parsing.preprocessing.strip_punctuation2(text)
        spell = SpellChecker()
        misspelled = text.split()
        wordnet_lemmatizer = WordNetLemmatizer()
        for i in range(len(misspelled)):
             word = spell.correction(misspelled[i])
             misspelled[i]=word
             misspelled[i] = wordnet_lemmatizer.lemmatize(misspelled[i], pos="v")
             misspelled[i] = wordnet_lemmatizer.lemmatize(misspelled[i], pos="n")
        text = " ".join(misspelled)
        filtered_words = [word for word in text.split() if word not in stops]
        text = " ".join(filtered_words)
        text = gensim.corpora.textcorpus.strip_multiple_whitespaces(text)
        return text