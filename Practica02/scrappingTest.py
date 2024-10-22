#!/usr/bin/env python
# coding: utf-8

import feedparser
import pandas as pd
import os
import subprocess


columns = ['Source', 'Title', 'Content', 'Section', 'URL', 'Date']
csv_file = 'raw_data_corpus.csv'

if not os.path.exists(csv_file):
    df = pd.DataFrame(columns = columns)
    df.to_csv(csv_file, sep='\t', index=False)
else:
    df = pd.read_csv(csv_file, sep='\t')


jsection_names = ['Deportes', 'Economía', 'Ciencia', 'Cultura']
esection_names = ['Economia','Ciencia']


jDeportes = feedparser.parse('https://www.jornada.com.mx/rss/deportes.xml?v=1')
jEconomia = feedparser.parse('https://www.jornada.com.mx/rss/economia.xml?v=1')
jCiencia = feedparser.parse('https://www.jornada.com.mx/rss/ciencias.xml?v=1')
jCultura = feedparser.parse('https://www.jornada.com.mx/rss/cultura.xml?v=1')

eEconomia = feedparser.parse('https://expansion.mx/rss/economia')
eCiencia = feedparser.parse('https://expansion.mx/rss/tecnologia')


jSections= [jDeportes,jEconomia,jCiencia,jCultura]
eSections = [eEconomia,eCiencia]


existing_news = df['Title'].tolist()


jnew_rows = []
enew_rows=[]
newEntries = 0


for i, section in enumerate(jSections):
    for entry in section.entries:
        if entry.title not in existing_news:
            new_row = {
                'Source': entry.get('Source', 'La Jornada'),
                'Title' : entry.title,
                'Content': entry.description,
                'Section': jsection_names[i],
                'URL': entry.link,
                'Date': entry.published 
            }
            newEntries += 1
            jnew_rows.append(new_row)

for i, section in enumerate(eSections):
    for entry in section.entries:
        if entry.title not in existing_news:
            new_row = {
                'Source': entry.get('Source', 'Expansión'),
                'Title': entry.title,
                'Content': entry.description,
                'Section': esection_names[i],
                'URL': entry.link,
                'Date': entry.published
            }
            newEntries += 1
            enew_rows.append(new_row)


jnew_data = pd.DataFrame(jnew_rows, columns = columns)
enew_data = pd.DataFrame(enew_rows, columns = columns)

df = pd.concat([df, jnew_data], ignore_index=True)
df = pd.concat([df, enew_data], ignore_index=True)

df.to_csv(csv_file, sep = "\t", index = False)


print("new entries: " +  str(newEntries))

subprocess.run(["python", "date.py"])