#!/usr/bin/env python
# coding: utf-8

import re
import pandas as pd
from collections import Counter

tweets = pd.read_csv("tweets.csv")

column = tweets['text']

print(tweets)

hashtagRe = r'#\w+'

hashtagCounter = 0
hashtagAcc = {}

for idx, row in enumerate(column): 
    matches = re.findall(hashtagRe, row)
    if matches:
        print(idx +1, matches)
    for match in matches:
        hashtagCounter += 1
        if match in hashtagAcc:
            hashtagAcc[match] += 1
        else:
            hashtagAcc[match] = 1

sortedHD = dict(sorted(hashtagAcc.items(), key=lambda item: item[1], reverse=True))
print(sortedHD)

print(hashtagCounter)

userRe = r'@[A-Za-z_]\w{0,14}'

userCounter = 0
userAcc = {}

for idx, row in enumerate(column):
    matches = re.findall(userRe, row)
    if matches:
        print(idx + 1, matches)
    for match in matches:
        userCounter += 1
        if match in userAcc:
            userAcc[match] += 1
        else:
            userAcc[match] = 1

sortedUD = dict(sorted(userAcc.items(), key=lambda item: item[1], reverse=True))
print(sortedUD)

print(userCounter)

urlRe = re.compile(
    r'https?:\/\/(?:(?:(?:[A-Za-z0-9-])+\.))+(?:[A-Za-z]){2,}(?::\d{1,5})?(?:(?:\/(?:[\w\-\.~%])+)*)?(?:\?(?:(?:[\w\-\.~%])+=(?:[\w\-\.~%])*(?:&(?:[\w\-\.~%])+=(?:[\w\-\.~%])*)*)?)?(?:#(?:[\w\-\.~%\/\+\!\@\(\)\[\]\{\}]+))??')

urlCounter = 0
urlAcc = {}

for idx, row in enumerate(column): 
    matches = re.findall(urlRe, row)
    if matches:
        print(idx + 1, matches)
    for match in matches:
        urlCounter += 1
        if match in urlAcc:
            urlAcc[match] += 1
        else:
            urlAcc[match] = 1

sortedURLD = dict(sorted(urlAcc.items(), key=lambda item: item[1], reverse=True))
print(sortedURLD)

print(urlCounter)

horaRe = r'(?:(?:(?:\d(?<=[0-2])\d)|(?:[1-9]))(?::(?:[0-5]\d))+?(?:am|AM|pm|PM|hrs?|HRS?|horas?)?)|(?:\d(?<=[0-2])\d|[1-9])(?::[0-5]\d)?\s?(?:am|AM|pm|PM|hrs?|HRS?horas?)'

horaCounter = 0
horaAcc = {}

for idx, row in enumerate(column): 
    matches = re.findall(horaRe, row)
    if matches:
        print(idx + 1, matches)
    for match in matches:   
        horaCounter += 1
        if match in horaAcc:
            horaAcc[match] += 1
        else:
            horaAcc[match] = 1

sortedHoraD = dict(sorted(horaAcc.items(), key=lambda item: item[1], reverse=True))
print(sortedHoraD)

print(horaCounter)

emoticonoRe = r'(?:>?(?::|;)-?(?:\)|\(|D|p|P|0|\\|O|\|))|(?:(?:x|X)D)'

emoticonoCounter = 0
emoticonoAcc = {}

for idx, row in enumerate(column): 
    matches = re.findall(emoticonoRe, row)
    if matches:
        print(idx + 1, matches)
    for match in matches:   
        emoticonoCounter += 1
        if match in emoticonoAcc:
            emoticonoAcc[match] += 1
        else:
            emoticonoAcc[match] = 1

sortedEmoticonoD = dict(sorted(emoticonoAcc.items(), key=lambda item: item[1], reverse=True))
print(sortedEmoticonoD)

print(emoticonoCounter)

emojiRe = re.compile(
    r'(?:'
    r'[\U0001F600-\U0001F64F'  
    r'\U0001F300-\U0001F5FF'   
    r'\U0001F680-\U0001F6FF'   
    r'\U0001F1E0-\U0001F1FF'   
    r'\U00002702-\U000027B0'   
    r'\U000024C2-\U0001F251'   
    r'\U0001F900-\U0001F9FF'   
    r'\U0001FA70-\U0001FAFF'   
    r'\U00002600-\U000026FF'   
    r'\U00002B50-\U00002B55'   
    r']|'
    r'\U0001F1E6-\U0001F1FF'   
    r')(?![\uFE0F\u200D\u1F3FB-\u1F3FF])'  
)

emojisCounter = 0
emojisAcc = {}

for idx, row in enumerate(column): 
    matches = re.findall(emojiRe, row)
    if matches:
        print(idx + 1, matches)
    for match in matches:   
        emojisCounter += 1
        if match in emojisAcc:
            emojisAcc[match] += 1
        else:
            emojisAcc[match] = 1

sortedEmojiD = dict(sorted(emojisAcc.items(), key=lambda item: item[1], reverse=True))
print(sortedEmojiD)

print(emojisCounter)

diccionarios = [sortedHD, sortedUD, sortedURLD, sortedHoraD, sortedEmoticonoD, sortedEmojiD]

tabla = {
    'String': ['Hashtags', 'Usuarios', 'URLs', 'Horas', 'Emoticonos', 'Emojis'],
    'Total Frequency': [hashtagCounter, userCounter, urlCounter, horaCounter, emoticonoCounter, emojisCounter],
    'Top 10': diccionarios
}

df = pd.DataFrame(tabla)
df['Top 10'] = df['Top 10'].apply(lambda x: '\n'.join([f'{repr(k)}: {v}' for k, v in list(x.items())[:10]]))

df.to_csv('tablaPract1.csv', index=False, encoding='utf-8')
