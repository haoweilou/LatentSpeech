import nltk
from nltk.corpus import cmudict
# Example usage
arpabet_to_ipa = {
    'AA': 'ɑ', 'AE': 'æ', 'AH': 'ʌ', 'AO': 'ɔ', 'AW': 'aʊ', 
    'AY': 'aɪ', 'B': 'b', 'CH': 'tʃ', 'D': 'd', 'DH': 'ð',
    'EH': 'ɛ', 'ER': 'ɝ', 'EY': 'eɪ', 'F': 'f', 'G': 'g',
    'HH': 'h', 'IH': 'ɪ', 'IY': 'i', 'JH': 'dʒ', 'K': 'k',
    'L': 'l', 'M': 'm', 'N': 'n', 'NG': 'ŋ', 'OW': 'oʊ',
    'OY': 'ɔɪ', 'P': 'p', 'R': 'ɹ', 'S': 's', 'SH': 'ʃ',
    'T': 't', 'TH': 'θ', 'UH': 'ʊ', 'UW': 'u', 'V': 'v',
    'W': 'w', 'Y': 'j', 'Z': 'z', 'ZH': 'ʒ'
}
# print(len(arpabet_to_ipa.keys()))
pronouncing_dict = cmudict.dict()

def word_to_phoneme(word):
    word_lower = word.lower()  # CMU Dict uses lowercase words
    if word_lower in pronouncing_dict:
        # Return the first pronunciation for simplicity
        return pronouncing_dict[word_lower][0]
    else:
        return None  # Word not found in the dictionary

def phoneme_to_ipa(phonemes):
    return [arpabet_to_ipa.get(p) for p in phonemes]

def pho_stress_split(phonemes):
    stress = []
    phoneme = []
    for pho in phonemes:
        last = pho[-1]
        if str.isnumeric(last): 
            stress.append(last)
            phoneme.append(pho[:-1])
        else: 
            stress.append(0)
            phoneme.append(pho)
    return phoneme,stress

def word_to_ipa(word):
    phonemes = word_to_phoneme(word)
    phoneme, stress = pho_stress_split(phonemes)
    ipa_phoneme = phoneme_to_ipa(phoneme)
    return ipa_phoneme, stress

def normalize_sentence(sentence):
    words = sentence.split(" ")
    output = []
    for word in words: 
        if word in pronouncing_dict:
            output.append(word)
        else: 
            output += split_into_words(word,pronouncing_dict)
    return " ".join(output)
    


def split_into_words(word, word_dictionary):
    word = word.lower()
    result = []
    i = 0
    while i < len(word):
        found = False
        for j in range(len(word), i, -1): 
            segment = word[i:j]
            if segment in word_dictionary:
                result.append(segment)
                i = j - 1
                found = True
                break
        if not found:  
            result.append(word[i])
        i += 1
    return result


def english_sentence_to_ipa(sentence):
    sentence = normalize_sentence(sentence)
    #english
    ipa_phonemes = []
    stresses = []
    
    for word in sentence.split(" "):
        ipa_phoneme, stress = word_to_ipa(word)
        ipa_phonemes += ipa_phoneme
        stresses += stress
    return ipa_phonemes, stresses
        
    
# print(english_sentence_to_ipa("woodcutters"))

#phoneme, number of ipa phoneme
#stress, 3, 0: no stress, 1 for primary stress, 2 for non-stress, mostly applied to vowel
#tone

pinyin_to_ipa = {
    "b": "p", "p": "pʰ", "m": "m", "f": "f", 
    "d": "t", "t": "tʰ", "n": "n", "l": "l",
    "g": "k", "k": "kʰ", "h": "x",
    "j": "tɕ", "q": "tɕʰ", "x": "ɕ",
    "zh": "ʈʂ", "ch": "ʈʂʰ", "sh": "ʂ", "r": "ɻ",
    "z": "ts", "c": "tsʰ", "s": "s",
    "w": "w", "y": "j",
    "a": "a", "o": "o", "e": "ɤ", "i": "i", "u": "u", "v": "y",
    "ai": "aɪ", "ei": "eɪ", "ao": "aʊ", "ou": "oʊ",
    "an": "an", "en": "ən", "ang": "ɑŋ", "eng": "əŋ", "ong": "ʊŋ",
    "er": "ɚ",
    "ia": "ja", "iao": "jaʊ", "ie": "je", "iu": "joʊ",
    "ian": "jan", "in": "in", "iang": "jɑŋ", "ing": "iŋ",
    "ua": "wa", "uo": "wo", "uai": "waɪ", "ui": "weɪ",
    "uan": "wan", "un": "wən", "uang": "wɑŋ", "ue": "yɛ", "van": "yan", "vn": "yn",
    "ve":"yɛ"
}

from pypinyin import pinyin, lazy_pinyin, Style
def hanzi_to_pinyin(hanzi):
    return [syllable[0] for syllable in pinyin(hanzi, style=Style.TONE3)]
                                               
initials = [
    "b", "p", "m", "f", "d", "t", "n", "l", "g", "k", "h",
    "j", "q", "x", "zh", "ch", "sh", "r", "z", "c", "s", "w", "y"
]
finals = [
    "a", "o", "e", "i", "u", "v",
    "ai", "ei", "ao", "ou",
    "an", "en", "ang", "eng", "ong",
    "ia", "ie", "iao", "iu", "ian", "in", "iang", "ing",
    "ua", "uo", "uai", "ui", "uan", "un", "uang",
    "ve", "van", "vn",
    "er"
]
def pinyin_to_phoneme(syllable):
    initial = None
    if syllable[:2] in initials: 
        initial = syllable[:2]
        final = syllable[2:]
    elif syllable[:1] in initials:
        initial = syllable[:1]
        final = syllable[1:]
    else: 
        final = syllable
    tone = final[-1] if final[-1].isdigit() else "0"
    if final[-1].isdigit(): final = final[:-1]
    if final == "iong": 
        final = ["i","ong"]
    else: 
        final = [final]
    if "" in final: final.remove("")
    if len(final) == 0: 
        phoneme = [initial]
        tones = [int(tone)]
    else: 
        phoneme = [initial]+final if initial is not None else final
        tones = [0]*(len(phoneme)-1)+[int(tone)]
    return phoneme, tones

def pinyin_to_ipa_phoneme(syllable):
    phoneme, tones = pinyin_to_phoneme(syllable)
    ipa_phoneme = [pinyin_to_ipa.get(p) for p in phoneme]
    return ipa_phoneme, tones

def pinyin_sentence_to_ipa(pinyin_sentence):
    ipa_phonemes = []
    tones = []
    for pinyin in pinyin_sentence:
        ipa_phoneme, tone = pinyin_to_ipa_phoneme(pinyin)
        ipa_phonemes += ipa_phoneme
        tones += tone
    return ipa_phonemes, tones

def mandarin_chinese_to_ipa(sentence):
    pinyin_sentence = hanzi_to_pinyin(sentence)
    return pinyin_sentence_to_ipa(pinyin_sentence)
    
def ipa_to_idx(ipa_phonemes):
    return  [ipa_pho_dict[i] for i in ipa_phonemes]   

all_ipa_phoneme = list(pinyin_to_ipa.values())+list(arpabet_to_ipa.values())
all_ipa_phoneme = sorted(list(set(all_ipa_phoneme)))
all_ipa_phoneme = ["EMPTY"] + all_ipa_phoneme
ipa_pho_dict = {k:i for i,k  in enumerate(all_ipa_phoneme)}

# print(len(ipa_pho_dict))

# ipa_pho, tone = mandarin_chinese_to_ipa("嗯八卦新闻总是很多")
# print(ipa_pho,tone)
# ipd_idx = [ipa_pho_dict[i] for i in ipa_pho]
# tone = tone 


    
