import pandas as pd
import warnings
import spacy
import pickle
from os import path

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

from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GridSearchCV        # https://scikit-learn.org/dev/modules/generated/sklearn.model_selection.GridSearchCV.html
from sklearn.exceptions import UndefinedMetricWarning

nlp = spacy.load('es_core_news_sm')
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


# NORMALIZACIONES
def stopWords(features):
    doc = nlp(features)
    tokens = [token.text for token in doc if not token.pos_ in ["DET", "ADP", "CCONJ", "SCONJ", "PRON"]]
    return " ".join(tokens)

def lemmatization(features):
    doc = nlp(features)
    tokens = [token.lemma_ for token in doc]
    return " ".join(tokens)

def lemmatizationStopWords(features):
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

    normalization_versions = [stopWords, lemmatization , lemmatizationStopWords]
    for version in normalization_versions:
        normalized_features = train_sf['Features'].apply(version)
        norm_df = pd.DataFrame({
            'Features': normalized_features,
            'Target': train_sf['Target']
        })
        training_normalizations.append(norm_df)
    return training_normalizations

param_grid = {
    MultinomialNB: {'classifier__alpha': [0.1, 0.5, 1.0]},
    LogisticRegression: {'classifier__C': [0.01, 0.1, 1, 10]},
    RidgeClassifier: {'classifier__alpha': [0.1, 1.0, 10.0]},
    NearestCentroid: {'classifier__metric': ['euclidean', 'manhattan']},
    LinearSVC: {'classifier__C': [0.01, 0.1, 1, 10]},
    MLPClassifier: {
        'classifier__hidden_layer_sizes': [(50,), (100,), (50, 50)],
        'classifier__alpha': [0.0001, 0.001]
    }
}

def classify(training_sets, test_set):
    normalizations = ['No normalization', 'Stop Words', 'Lemmatization', 'Stop Words + Lemmatization']
    representations = [CountVectorizer(), CountVectorizer(binary=True), TfidfVectorizer()]
    dimentionality_reductions = [None, TruncatedSVD(1000), TruncatedSVD(500), TruncatedSVD(300)]
    classifiers = [MultinomialNB(), LogisticRegression(max_iter=1000), RidgeClassifier(), NearestCentroid(), LinearSVC(max_iter=5000), MLPClassifier(max_iter=1000)]

    x_test = test_set['Features']
    y_test = test_set['Target']
    
    for classifier in classifiers:
        classifier_name = classifier.__class__.__name__
        print('\t=====' + str(classifier_name) + '=====')
        
        for reduction in dimentionality_reductions:
            # Omite la combinación de MultinomialNB con cualquier TruncatedSVD
            if isinstance(classifier, MultinomialNB) and reduction is not None:
                continue  # Salta esta combinación
            
            for i, training_set in enumerate(training_sets):
                x_train = training_set['Features']
                y_train = training_set['Target']
                
                for representation in representations:
                    # Configura el pipeline dinámicamente para incluir o no el paso de reducción de dimensionalidad
                    if reduction is None:
                        pipe = Pipeline([('text_representation', representation), ('classifier', classifier)])
                    else:
                        pipe = Pipeline([('text_representation', representation), ('dimentionality_reduction', reduction), ('classifier', classifier)])

                    param_search = param_grid.get(type(classifier), {})
                    grid = GridSearchCV(pipe, param_search, cv=5, scoring='f1_macro', n_jobs=-1)

                    grid.fit(x_train, y_train)
                    best_model = grid.best_estimator_

                    print(normalizations[i])
                    print(representation)
                    print(reduction)
                    print(grid.best_params_)

                    y_pred = best_model.predict(x_test)
                    print(classification_report(y_test, y_pred, zero_division=0))
        
        print('\n\n')





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
