import os
import re
import string
import pickle
import langid
import matplotlib.style
import matplotlib.pyplot as plt
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from typing import List
from dotenv import load_dotenv
from spellchecker import SpellChecker
from jpype import JClass, JString, getDefaultJVMPath, shutdownJVM, startJVM, java, isJVMStarted


def beautify_turkish(data):
    ZEMBEREK_PATH = r'data/zemberek-full.jar'
    DATA_PATH = "data"
    if isJVMStarted() is False:
        startJVM(getDefaultJVMPath(), '-ea', '-Djava.class.path=%s' % ZEMBEREK_PATH)
    TurkishMorphology = JClass('zemberek.morphology.TurkishMorphology')
    TurkishSentenceNormalizer: JClass = JClass(
        "zemberek.normalization.TurkishSentenceNormalizer"
    )
    TurkishTokenizer: JClass = JClass("zemberek.tokenization.TurkishTokenizer")
    Paths: JClass = JClass("java.nio.file.Paths")
    morphology = TurkishMorphology.createWithDefaults()
    normalizer = TurkishSentenceNormalizer(
        TurkishMorphology.createWithDefaults(),
        Paths.get(str(os.path.join(DATA_PATH, "normalization"))),
        Paths.get(str(os.path.join(DATA_PATH, "lm", "lm.2gram.slm"))),
    )

    tokenizer = TurkishTokenizer.DEFAULT
    url_pattern = re.compile(r'https?://\S+|www\.\S+')

    def lemmatize(words):
        if words:
            analysis: java.util.ArrayList = (
                morphology.analyzeAndDisambiguate(words).bestAnalysis()
            )
            pos: List[str] = []
            for i, analysis in enumerate(analysis, start=1):
                f'\nAnalysis {i}: {analysis}',
                f'\nPrimary POS {i}: {analysis.getPos()}'
                f'\nPrimary POS (Short Form) {i}: {analysis.getPos().shortForm}'
                if str(analysis.getLemmas()[0]) != "UNK":
                    pos.append(
                        f'{str(analysis.getLemmas()[0])}'
                    )
                else:
                    pos.append(f'{str(analysis.surfaceForm())}')

            return " ".join(pos)
        else:
            return words

    def tokenize(text):
        tokens = []
        for i, token in enumerate(tokenizer.tokenizeToStrings(JString(text))):
            tokens.append(str(token))
        return tokens

    stop_words = set(stopwords.words("turkish"))
    data["text"] = data["text"].apply(lambda x: url_pattern.sub(r'', x))
    data["text"] = data["text"].apply(lambda x: str(normalizer.normalize(JString(x))))
    data["text"] = data["text"].apply(lambda x: "".join([i for i in x if i not in string.punctuation]))
    data["text"] = data["text"].apply(lambda x: "".join([i for i in x if not i.isdigit()]))
    data["text"] = data["text"].apply(lambda x: tokenize(x))
    data["text"] = data["text"].apply(lambda x: [i for i in x if i not in stop_words])
    data["text"] = data["text"].apply(lambda x: [lemmatize(i) for i in x])
    data["text"] = data["text"].apply(lambda x: " ".join([i for i in x]))
    data["text"] = data["text"].apply(lambda x: x.replace('"', ''))
    data["text"] = data["text"].apply(lambda x: x.replace("'", ''))
    data['text'] = data['text'].str.replace('\d+', '')

    return data


def beautify_english(data):
    spell = SpellChecker()
    stop_words = stopwords.words("english")
    tokenizer = RegexpTokenizer(r"\w+")
    url_pattern = re.compile(r'https?://\S+|www\.\S+')

    def lemmatize(text):

        result = []
        wordnet = WordNetLemmatizer()
        for token, tag in nltk.pos_tag(text):
            pos = tag[0].lower()

            if pos not in ['a', 'r', 'n', 'v']:
                pos = 'n'

            result.append(wordnet.lemmatize(token, pos))

        return result

    data["text"] = data["text"].apply(lambda x: str(x))
    data["text"] = data["text"].apply(lambda x: " ".join(x.split()))
    data["text"] = data["text"].apply(lambda x: url_pattern.sub(r'', x))
    data['text'] = data['text'].str.replace('\d+', '')
    data["text"] = data["text"].apply(lambda x: nltk.word_tokenize(x))
    data["text"] = data["text"].apply(lambda x: [spell.correction(word) for word in x])
    data["text"] = data["text"].apply(lambda x: [word for word in x if word not in stop_words])
    data["text"] = data["text"].apply(lambda x: [word for word in x if isinstance(word, str)])
    data["text"] = data["text"].apply(lambda x: tokenizer.tokenize(' '.join(x)))
    data["text"] = data["text"].apply(lemmatize)
    data["text"] = data["text"].apply(lambda x: ' '.join(x))
    data["text"] = data["text"].apply(lambda x: x.replace('"', ''))
    data["text"] = data["text"].apply(lambda x: x.replace("'", ''))

    return data


def english_turkish_split(data):
    data["language"] = data["text"].apply(lambda x: langid.classify(x)[0])
    english_data = data[data["language"] == "en"]
    turkish_data = data[data["language"] == "tr"]
    return english_data, turkish_data


data = pd.read_csv("twitter_validation_eng.csv", encoding="utf-8", usecols=["id", "label", "text"], index_col="id")
data["text"] = data["text"].str.lower()

data["text"] = data["text"].apply(lambda x: x.replace("gözen holding", ""))

data = beautify_english(data)

english_data, turkish_data = english_turkish_split(data)

english_data = beautify_english(english_data)
turkish_data = beautify_turkish(turkish_data)

data = pd.concat([english_data, turkish_data])
data.to_csv("processed_tweets.csv", encoding="utf-8")
