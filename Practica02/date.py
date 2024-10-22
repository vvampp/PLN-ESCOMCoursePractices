import pandas as pd
import re

source_file = 'raw_data_corpus.csv'

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

try:
    sf = pd.read_csv(source_file, sep='\t')
except:
    print("An exception ocurred")

for i,row in sf.iterrows():
    match = re.search(regex, str(row.iloc[5]))
    if match:
        date  = match.group()
        for month, number in month_dict.items():
            if month in date:
                date = date.replace(month,number)
                date = date.replace(" ", "/")
                break
        sf.at[i,sf.columns[5]] = date

sf.to_csv(source_file, sep="\t", index=False)
    


