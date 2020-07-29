# -*- coding: utf-8 -*-
"""
Created on Tue Jun 16 18:43:11 2020

@author: maestro
"""
import re
import nltk

from nltk.stem.snowball import EnglishStemmer
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


stemmer = EnglishStemmer()
# REPLACE_BY_SPACE_RE = re.compile("[{}\[\]\|@,;\.'()]")
NUMBER = re.compile('[0-9]+')
STOPWORDS = {
    'a', 'an', 'the', "to",  'i', 'you', 'he', 'she', 'it', 'we', 'they'
},


def get_wordnet_pos(word):
    """Map POS tag to first character lemmatize() accepts"""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV
    }
    return tag_dict.get(tag, wordnet.NOUN)  # NOUN as default


def lemming(text):
    lemmatizer = WordNetLemmatizer()
    return [
        lemmatizer.lemmatize(word, get_wordnet_pos(word))
        for word in word_tokenize(text)
    ]


def fix_appends(text):
#    text = re.sub('(^|\s+)not\s+', ' not_', text)
#     text = re.sub('(^|\s+)no\s+', ' no_', text)
    return text


def clean_text(text):
    t = text.lower()  # lowercase text
    t = re.sub(' +', ' ', t)  # spaces to one space
    t = " ".join(lemming(t))
    t = ' '.join(stemmer.stem(word) for word in t.split())
    t = ' '.join(word for word in t.split() if word not in STOPWORDS)
    t = fix_appends(t)
    t = re.sub(' +', ' ', t)
    return t


fixes = {
    "are n't": " are not ",
     "aren 't": " are not ",
     "can 't": " can not ",
     "could n't": " could not " ,
     "couldn 't": " could not ",
     "did n't": " did not ",
     "didn 't": " did not ",
     "do n't": " do not ",
     "does n't": " does not ",
     "doesn 't": " does not ",
     "don 't": " do not ",
     "dosn 't": " does not ",
     "has n't": " has not ",
     "hasn 't": " has not ",
     "have n't": " have not ",
     "haven 't": " have not ",
     "is n't": " is not ",
     "isn 't": " is not ",
     "would't": " would not ",
     "sdon 't": " does not ",
     "should n't": " should not ",
     "was n't": " was not ",
     "wasn 't": " was not ",
     "were n't":  " were not ",
     "wo n't": " will not ",
     "won 't": " will not ",
     "would n't": " would not ",
     "wouldn 't": " would not ",
     "n't": " not ",
     "'t": " not ",
     'cant ': " can not ",
     'didnt ': " did not ",
     'doesnt ': " does not ",
     'dont ': " do not ",
     'havent ': " have not ",
     'wouldnt ': " would not ",
     "'d": " would ",
     "it 's": " it is ",
        "it's": " it is ",
     "i 'm": " i am ",
     "'re": " are ",
     "'s": " ",
     "\-+": " ",
     # no parenthesis () 
     "((^|\s)+(\s*[$.,-:/+ %#@*~']*[0-9]+[$.,-:/+ %#@*~']*)+ )": " ", 
     "\s+": " "
     }


def fix(texts):
    ts  = []
    mask = [False]*len(texts) # mask for corrections (dfefault: no corrections [all False])
    for i, t in enumerate(texts):
        for f in fixes.keys():
            match = re.findall("(\s*"+f+"\s*)+", t)
            if len(match)>0:
#                 print(match)
                mask[i]=True
                for m in match:
                    if isinstance(m, type("str")):
                        t= re.sub(m,fixes[f],  t)
                    else: t= re.sub(m[0],fixes[f],  t)
        ts.append(t)
    return ts,mask


def clean_and_fix(texts):
    return [clean_text(t) for t in fix(texts)[0]]

