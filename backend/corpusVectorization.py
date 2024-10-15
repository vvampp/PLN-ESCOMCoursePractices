from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import os.path
import pickle
from itertools import zip_longest



# Unigramas
def freqUniVectorize(corpus):
    vector = CountVectorizer()
    X = vector.fit_transform(corpus)
    return vector, X

def oneHotUniVectorize(corpus):
    vector = CountVectorizer(binary=True)
    X = vector.fit_transform(corpus)
    return vector, X

def tfidfUniVectorize(corpus):
    vector = TfidfVectorizer()
    X = vector.fit_transform(corpus)
    return vector, X

# Bigramas
def freqBiVectorize(corpus):
    vector = CountVectorizer(ngram_range=(2,2))
    X = vector.fit_transform(corpus)
    return vector, X

def oneHotBiVectorize(corpus):
    vector = CountVectorizer(binary=True,ngram_range=(2,2))
    X = vector.fit_transform(corpus)
    return vector, X

def tfidfBiVectorize(corpus):
    vector = TfidfVectorizer(ngram_range=(2,2))
    X = vector.fit_transform(corpus)
    return vector, X




def vectorizeAll(source_file):

    title = source_file['Title'].tolist()
    content = source_file['Content'].tolist()
    tyc =  [itemT + " " + (itemC if itemC is not None else "") for itemT,itemC in zip_longest(title,content, fillvalue="")]

    corpuses = {'title': title, 'content': content, 'tyc': tyc}
    ngrams = ['uni','bi']
    vectorization_methods = {
        'freq': {'uni': freqUniVectorize, 'bi': freqBiVectorize},
        'oneHot': {'uni': oneHotUniVectorize, 'bi': oneHotBiVectorize},
        'tfidf': {'uni': tfidfUniVectorize, 'bi': tfidfBiVectorize},
    }

    vectorizedOutput_folder = os.path.join(os.getcwd(), 'vectorized_data')
    os.makedirs(vectorizedOutput_folder, exist_ok=True)

    vectorOutputFolder = os.path.join(os.getcwd(), 'vectorizers')
    os.makedirs(vectorOutputFolder, exist_ok=True)

    for corpus_name, corpus in corpuses.items():
        for  ngram in ngrams:
            for method_name, methods in vectorization_methods.items():

                vectorizer, vectorized_data = methods[ngram](corpus)

                vectorizedDataFilename = f'{corpus_name}_{method_name}_{ngram}.pkl'
                vectorizerFilename = f'{corpus_name}_{method_name}_{ngram}_vectorizer.pkl'

                vectorizedDataFilepath = os.path.join(vectorizedOutput_folder, vectorizedDataFilename)
                vectorizerFilepath = os.path.join(vectorOutputFolder, vectorizerFilename)

                with open(vectorizedDataFilepath, 'wb') as file:
                    pickle.dump(vectorized_data, file)

                with open(vectorizerFilepath, 'wb') as file:
                    pickle.dump(vectorizer,file)
                
                print(f'Vercotizacion guardada: {vectorizedDataFilename}')
                print(f'Vector guardado: {vectorizerFilename}')

