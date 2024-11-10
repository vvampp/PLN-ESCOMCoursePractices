import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import TruncatedSVD

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.neighbors import NearestCentroid
from sklearn.svm import LinearSVC
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

def classify(training_sets, test_set):

    x_test = test_set['Features']
    y_test = test_set['Target']

    normalizations = ['No normalization', 'Stop Words', 'Lemmatization', 'Stop Words + Lemmatization']
    representations = [CountVectorizer(), CountVectorizer(binary=True), TfidfVectorizer()]

    for i,trainig_set in enumerate(training_sets):

        x_train = trainig_set['Features']
        y_train = trainig_set['Target']

        for representation in representations:
            print(normalizations[i])
            pipe = Pipeline([('text_representation', representation), ('classifier', MLPClassifier())])
            print(pipe)
            pipe.fit(x_train, y_train)
            print ('Total Features: ' + str(len(pipe['text_representation'].get_feature_names_out())))
            y_pred = pipe.predict(x_test)
            print(classification_report(y_test,y_pred))





def main():
    source_tsv_file = 'destilled_raw_data_corpus.tsv'

    try:
        sf = pd.read_csv(source_tsv_file, sep='\t')
    except:
        print("An exception occurred")
        return

    train_sf, test_sf = train_test_split(sf, test_size=0.2, random_state=21, shuffle=True, stratify=sf['Target'])

    # print("Distribución en Train set:\n", train_sf['Target'].value_counts(normalize=True))
    # print("Distribución en Test set:\n", test_sf['Target'].value_counts(normalize=True))



    training_normalizations = normalize(train_sf)

    classify(training_normalizations,test_sf)




if __name__ == "__main__":
    main()
