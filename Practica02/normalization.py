import spacy
import pandas as pd
import os

# CSV
source_csv_file = 'raw_data_corpus.csv'
destiny_csv_file = 'normalized_data_corpus.csv'
columns = ['Source', 'Title', 'Content', 'Section', 'URL', 'Date']

try:
    sf = pd.read_csv(source_csv_file, sep='\t')
except:
    print("An exception ocurred")


if not os.path.exists(destiny_csv_file):
    df = pd.DataFrame(columns=columns)
    df.to_csv(destiny_csv_file, sep='\t', index=False)
else:
    df = pd. read_csv(destiny_csv_file, sep = '\t')


new_rows = []

# normalización

nTitle = ""
nContent = ""

nlp = spacy.load('es_core_news_sm')


for i,row in sf.iterrows():

    title = nlp(str(row.iloc[1]))
    content = nlp(str(row.iloc[2])) 

    for token in title:
        if not token.pos_ in ["DET", "ADP", "CCONJ", "SCONJ", "PRON"]:
            nTitle = nTitle + token.lemma_ + " "
        

    for token in content:
        if not token.pos_ in ["DET", "ADP", "CCONJ", "SCONJ", "PRON"]:
            nContent = nContent + token.lemma_ + " "

    new_row = {
        'Source': row.iloc[0],
        'Title': nTitle,
        'Content': nContent,
        'Section': row.iloc[3],
        'URL': row.iloc[4],
        'Date': row.iloc[5]
    }
    new_rows.append(new_row)

    nTitle = ""
    nContent = ""


new_data = pd.DataFrame(new_rows, columns=columns)

df = pd.concat([df,new_data], ignore_index=True)

df.to_csv(destiny_csv_file, sep='\t', index = False);