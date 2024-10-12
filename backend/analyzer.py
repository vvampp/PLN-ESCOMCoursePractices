import os.path
import pickle
from itertools import zip_longest
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd


# Unigramas
def freqUniVectorize(test):
    freq_vectorizator_uni = CountVectorizer()
    Y = freq_vectorizator_uni.transform(test)
    return Y

def oneHotUniVectorize(test):
    oneHot_vectorizator_uni = CountVectorizer(binary=True)
    Y = oneHot_vectorizator_uni.transform(test)
    return Y

def tfidfUniVectorize(test):
    tfidf_vectorizator_uni = TfidfVectorizer()
    Y = tfidf_vectorizator_uni.transform(test)
    return Y

# Bigramas
def freqBiVectorize(test):
    freq_vectorizator_bi = CountVectorizer(ngram_range=(2,2))
    Y = freq_vectorizator_bi.transform(test)
    return Y

def oneHotBiVectorize(test):
    oneHot_vectorizator_bi = CountVectorizer(binary=True,ngram_range=(2,2))
    Y = oneHot_vectorizator_bi.transform(test)
    return Y

def tfidfBiVectorize(test):
    tfidf_vectorizator_bi = TfidfVectorizer(ngram_range=(2,2))
    Y = tfidf_vectorizator_bi.transform(test)
    return Y



def vectorizeTest(test_csv_file,compare_element, vector_type, feature_type):
    # alimentar funcion con archivo directamente (tests)
    # test_csv_file = 'prueba.csv'
    try:
        source_file = pd.read_csv(test_csv_file,sep='\t')
    except Exception as e:
        print(f"An exception ocurred: {e}")

    # valores para realizar tests
    # compare_element = 'tyc'
    # vector_type = 'freq'
    # feature_type = 'bi'
    
    # Recuperar el test con base en el elemento a comparar
    if(compare_element == 'titulo'):
        print("Comparar por titulo")
        test = source_file['Title'].tolist()

    elif(compare_element == 'contenido'):
        print("Comparar por contenido")
        test = source_file['Content'].tolist()
    else:
        print("Comparar por titulo y contenido")
        titletest = source_file['Title'].tolist()
        contenttest = source_file['Content'].tolist()
        test = [itemT + " " + (itemC if itemC is not None else "") for itemT, itemC in
                       zip_longest(titletest, contenttest, fillvalue="")]


    vectorizer_map = {
        ('freq', 'uni'): CountVectorizer(),
        ('onehot', 'uni'): CountVectorizer(binary=True),
        ('tfidf', 'uni'): TfidfVectorizer(),
        ('freq', 'bi'): CountVectorizer(ngram_range=(2, 2)),
        ('onehot', 'bi'): CountVectorizer(binary=True, ngram_range=(2, 2)),
        ('tfidf', 'bi'): TfidfVectorizer(ngram_range=(2, 2))
    }

    # Selección de la función de vectorización
    try:
        vectorizer = vectorizer_map[(vector_type, feature_type)]
    except KeyError:
        raise ValueError(f'Tipo de vectorización o característica no reconocidos: {vector_type}, {feature_type}')
    
    Y = vectorizer.fit_transform(test)

    output_folder = os.path.join(os.getcwd(), 'vectorized_test')    # si se prueba directamente, el folder se crea en el workning dir, y no en backend
    os.makedirs(output_folder, exist_ok=True)
    filename = f'test_{compare_element}_{vector_type}_{feature_type}.pkl'
    filepath = os.path.join(output_folder, filename)
    with open(filepath, 'wb') as file:
        pickle.dump(Y, file)
    print(f'Vectorización de test guardado: {filename}')

# Renombrar a vectorizeTest "main" para hacer pruebas
# if __name__ == "__main__":
#     main()


# QUIZÁS HAGA FALTA IMPEMENTAR QUE SE BORREN LOS CONTENIDOS DE LA CARPETA DE vectorized_test PARA NO DAR ERRORES
# A LA HORA DE APLICAR LA SIMILITUD COSENO (SELECCIONAR UNA ENTRE MUCHAS VECTORIZACIONES ANTERIORES)