import pandas as pd


def main():
    source_tsv_file = 'raw_data_corpus.tsv'
    destiny_tsv_file = 'destilled_raw_data_corpus.tsv'
    columns = ['Features','Target']

    try:
        sf = pd.read_csv(source_tsv_file, sep= '\t')
    except:
        print("An exception occurred") 
    

    rows = []

    for i,row in sf.iterrows():
        rows.append({
            "Features": str(row.iloc[1]) + " " + str(row.iloc[2]),
            "Target": str(row.iloc[3])
        })

    df = pd.DataFrame(rows, columns=columns)
    df.columns = df.columns.str.strip()
    df.to_csv(destiny_tsv_file,sep='\t', index=False)


if __name__ == "__main__":
    main()