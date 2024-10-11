import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
import os.path
import pickle
from itertools import zip_longest
import argparse



# Unigramas
def freqUniVectorize(corpus):
    freq_vectorizator_uni = CountVectorizer()
    X = freq_vectorizator_uni.fit_transform(corpus)
    return X

def oneHotUniVectorize(corpus):
    oneHot_vectorizator_uni = CountVectorizer(binary=True)
    X = oneHot_vectorizator_uni.fit_transform(corpus)
    return X

def tfidfUniVectorize(corpus):
    tfidf_vectorizator_uni = TfidfVectorizer()
    X = tfidf_vectorizator_uni.fit_transform(corpus)
    return X

# Bigramas
def freqBiVectorize(corpus):
    freq_vectorizator_bi = CountVectorizer(ngram_range=(2,2))
    X = freq_vectorizator_bi.fit_transform(corpus)
    return X

def oneHotBiVectorize(corpus):
    oneHot_vectorizator_bi = CountVectorizer(binary=True,ngram_range=(2,2))
    X = oneHot_vectorizator_bi.fit_transform(corpus)
    return X

def tfidfBiVectorize(corpus):
    tfidf_vectorizator_bi = TfidfVectorizer(ngram_range=(2,2))
    X = tfidf_vectorizator_bi.fit_transform(corpus)
    return X


def main():

    parser = argparse.ArgumentParser(description="Vectorización de Corpus Normalizado")
    parser.add_argument('file_path', type=str, help="Ruta del archivo CSV con el corpus normalizado (normalized_data_corpus.csv)")
    arguments = parser.parse_args()

    try:
        source_file = pd.read_csv(arguments.file_path, sep='\t')
    except:
        print("An excepction occurred")

    titleCorpus = source_file['Title'].tolist()
    contentCorpus = source_file['Content'].tolist()
    tandCcorpus =  [itemT + " " + (itemC if itemC is not None else "") for itemT,itemC in zip_longest(titleCorpus,contentCorpus, fillvalue="")]

    corpuses = {'titleCorpus': titleCorpus, 'contentCorpus': contentCorpus, 'tandCcorpus': tandCcorpus}
    ngrams = ['uni','bi']
    vectorization_methods = {
        'freq': {'uni': freqUniVectorize, 'bi': freqBiVectorize},
        'oneHot': {'uni': oneHotUniVectorize, 'bi': oneHotBiVectorize},
        'tfidf': {'uni': tfidfUniVectorize, 'bi': tfidfBiVectorize},
    }

    output_folder = os.path.join(os.getcwd(), 'vectorized_data')
    os.makedirs(output_folder, exist_ok=True)

    for corpus_name, corpus in corpuses.items():
        for  ngram in ngrams:
            for method_name, methods in vectorization_methods.items():

                vectorized_data = methods[ngram](corpus)

                filename = f'{corpus_name}_{method_name}_{ngram}.pkl'
                filepath = os.path.join(output_folder, filename)

                with open(filepath, 'wb') as file:
                    pickle.dump(vectorized_data, file)
                
                print(f'Archivo guardado: {filename}')




if __name__ == "__main__":
    main()