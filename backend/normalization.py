import spacy 
import pandas as pd
import os
import re

columns = ['Source', 'Title', 'Content', 'Section', 'URL', 'Date']

def normalizeDate(sf):
    month_dict = {
        "Jan": "01",
        "Feb": "02",
        "Mar": "03",
        "Apr": "04",
        "May": "05",
        "Jun": "06",
        "Jul": "07",
        "Aug": "08",
        "Sep": "09",
        "Oct": "10",
        "Nov": "11",
        "Dec": "12"
    }
    regex = r'\d\d?\s\w{3}\s\d{4}'
    match = re.search(regex,str(sf.iloc[0,5]))
    if match:
        date  = match.group()
        for month, number in month_dict.items():
            if month in date:
                date = date.replace(month,number)
                date = date.replace(" ", "/")
                break
        sf.at[0,sf.columns[5]] = date


def normalizeTC(sf,df):
    destiny_csv_file = 'normalized_test.csv'

    nTitle = ""
    nContent = ""

    nlp = spacy.load('es_core_news_sm')

    title = nlp(str(sf.iloc[0,1]))
    content = nlp(str(sf.iloc[0,2]))

    for token in title:
        if not token.pos_ in ["DET","ADP", "CCONJ", "SCONJ", "PRON"]:
            nTitle = nTitle + token.lemma_ + " "

    for token in content:
        if not token.pos_ in ["DET","ADP", "CCONJ", "SCONJ", "PRON"]:
            nContent = nContent + token.lemma_ + " "

    noralized_row = {
        'Source':sf.iloc[0,0], 
        'Title': nTitle,
        'Content': nContent,
        'Section': sf.iloc[0,3],
        'URL': sf.iloc[0,4],
        'Date': sf.iloc[0,5]
    }

    output_folder = os.path.join(os.getcwd(), 'normalized_test')
    os.makedirs(output_folder, exist_ok=True)
    filepath = os.path.join(output_folder, destiny_csv_file)

    new_data = pd.DataFrame([noralized_row], columns = columns)
    df = pd.concat([df,new_data], ignore_index=True)

    with open(filepath, 'wb') as file:
        df.to_csv(file, sep='\t', index = False)



def normalize (test_csv_file):
    try:
        sf = pd.read_csv(test_csv_file,sep='\t')
    except Exception as e:
        print(f"An exception ocurred: {e}")

    df = pd.DataFrame(columns=columns)

    normalizeDate(sf)
    normalizeTC(sf,df)   