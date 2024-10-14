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

    titleCorpus = source_file['Title'].tolist()
    contentCorpus = source_file['Content'].tolist()
    TandCcorpus =  [itemT + " " + (itemC if itemC is not None else "") for itemT,itemC in zip_longest(titleCorpus,contentCorpus, fillvalue="")]

    corpuses = {'titleCorpus': titleCorpus, 'contentCorpus': contentCorpus, 'TandCcorpus': TandCcorpus}
    ngrams = ['uni','bi']
    vectorization_methods = {
        'freq': {'uni': freqUniVectorize, 'bi': freqBiVectorize},
        'oneHot': {'uni': oneHotUniVectorize, 'bi': oneHotBiVectorize},
        'tfidf': {'uni': tfidfUniVectorize, 'bi': tfidfBiVectorize},
    }

    vectorizedOutput_folder = os.path.join(os.getcwd(), 'vectorized_data')
    os.makedirs(vectorizedOutput_folder, exist_ok=True)

    vectorOutputFolder = os.path.join(os.getcwd(), 'vectors')
    os.makedirs(vectorOutputFolder, exist_ok=True)

    for corpus_name, corpus in corpuses.items():
        for  ngram in ngrams:
            for method_name, methods in vectorization_methods.items():

                vector, vectorized_data = methods[ngram](corpus)
                print("SI ENTRAAA")

                vectorizedDataFilename = f'{corpus_name}_{method_name}_{ngram}.pkl'
                vectorFilename = f'{corpus_name}_{method_name}_{ngram}_{vector}.pkl'

                vectorizedDataFilepath = os.path.join(vectorizedOutput_folder, vectorizedDataFilename)
                vectorFilepath = os.path.join(vectorOutputFolder, vectorFilename)

                with open(vectorizedDataFilepath, 'wb') as file:
                    pickle.dump(vectorized_data, file)

                with open(vectorFilepath, 'wb') as file:
                    pickle.dump(vector,file)
                
                print(f'Vercotizacion guardada: {vectorizedDataFilename}')
                print(f'Vector guardado: {vectorFilename}')

