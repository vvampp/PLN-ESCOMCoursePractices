import pandas as pd
from sklearn.model_selection import train_test_split
import spacy

nlp = spacy.load('es_core_news_sm')

def stopWords(features):
    doc = nlp(features)
    tokens = [token.text for token in doc if not token.pos_ in ["DET", "ADP", "CCONJ", "SCONJ", "PRON"]]
    return " ".join(tokens)

def lemmatization(features):
    doc = nlp(features)
    tokens = [token.lemma_ for token in doc]
    return " ".join(tokens)

def lemmatiazionStopWords(features):
    doc = nlp(features)
    tokens = [token.lemma_ for token in doc if not token.pos_ in ["DET", "ADP", "CCONJ", "SCONJ", "PRON"]]
    return " ".join(tokens)

def main():
    source_tsv_file = 'destilled_raw_data_corpus.tsv'

    try:
        sf = pd.read_csv(source_tsv_file, sep='\t')
    except:
        print("An exception occurred")
        return


    train_sf, test_sf = train_test_split(sf, test_size=0.2, random_state=21, shuffle=True, stratify=sf['Target'])

    print("Distribución en Train set:\n", train_sf['Target'].value_counts(normalize=True))
    print("Distribución en Test set:\n", test_sf['Target'].value_counts(normalize=True))


    train_sf['Features'] = train_sf['Features'].apply(lemmatiazionStopWords)

    out = pd.DataFrame(train_sf)
    out.to_csv('output.tsv', sep='\t', index=False)

if __name__ == "__main__":
    main()
