import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import spacy

nlp = spacy.load('es_core_news_sm')


# NORMALIZACIONES
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


# NORMALIZAR
def normalize(train_sf):
    training_normalizations = []

    norm_df = pd.DataFrame({
        'Features': train_sf['Features'],
        'Target' : train_sf['Target']
    })
    training_normalizations.append(norm_df)

    normalization_versions = [stopWords, lemmatization , lemmatiazionStopWords]
    for version in normalization_versions:
        normalized_features = train_sf['Features'].apply(version)
        norm_df = pd.DataFrame({
            'Features': normalized_features,
            'Target': train_sf['Target']
        })
        training_normalizations.append(norm_df)
    return training_normalizations



def main():
    source_tsv_file = 'destilled_raw_data_corpus.tsv'

    try:
        sf = pd.read_csv(source_tsv_file, sep='\t')
    except:
        print("An exception occurred")
        return

    train_sf, test_sf = train_test_split(sf, test_size=0.2, random_state=21, shuffle=True, stratify=sf['Target'])
    # X_train = train_sf['Features']
    # X_train = train_sf['Features']
    # X_train = train_sf['Features']
    # X_train = train_sf['Features']


    # print("Distribución en Train set:\n", train_sf['Target'].value_counts(normalize=True))
    # print("Distribución en Test set:\n", test_sf['Target'].value_counts(normalize=True))

    training_normalizations = normalize(train_sf)

    outputs = ['RAW.tsv','SW.tsv','LMT.tsv','BOTH.tsv']

    for i,normalization in enumerate(training_normalizations):
        out = pd.DataFrame(normalization)
        out.to_csv(outputs[i], sep='\t', index=False)

    # pipe = Pipeline([('text_representation', TfidfVectorizer()), ('classifier', MultinomialNB())])
    # print(pipe)
    # pipe.fit(training_normalizations[3]['Features'], training_normalizations[3]['Target'])
    # print (len(pipe['text_representation'].get_feature_names_out()))
    # y_pred = pipe.predict(test_sf['Features'])
    # print(classification_report(test_sf['Target'],y_pred))


if __name__ == "__main__":
    main()
