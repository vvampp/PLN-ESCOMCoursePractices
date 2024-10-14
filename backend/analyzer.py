import os.path
import pickle
from itertools import zip_longest
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import math


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

def cosine(x, y):
	val = sum(x[index] * y[index] for index in range(len(x)))
	sr_x = math.sqrt(sum(x_val**2 for x_val in x))
	sr_y = math.sqrt(sum(y_val**2 for y_val in y))
	res = val/(sr_x*sr_y)
	return (res)

def cosine_similarity(test_vector, vector_type, feature_type, compare_element):
    # Recuperar el test
    try:
        with open(test_vector, 'rb') as file:
            test = pickle.load(file)
    except Exception as e:
        print(f"An exception ocurred: {e}")


    # Recuperar el corpus con base en el elemento a comparar
    if (compare_element == 'titulo'):
        print("Comparar por titulo")
    elif (compare_element == 'contenido'):
        print("Comparar por contenido")
    else:
        print("Comparar por titulo y contenido")

def main():
    # alimentar funcion con archivo directamente (tests)
    test_txt_file = 'test.txt'

    try:
        with open(test_txt_file, 'r') as file:
            test_file_content = file.read().rstrip() 
            test = list(test_file_content.split(" "))
            print(test)
    except Exception as e:
        print(f"An exception ocurred: {e}")

    # valores para realizar tests
    compare_element = 'tyc'
    vector_type = 'freq'
    feature_type = 'bi'
    
    # Recuperar el test con base en el elemento a comparar
    # NO NECESARIO SI EL FORMATO DE TEST.TXT SOLO CONTIENE EL CONTENIDO DE LA NOTICIA

    # print("Recuperando test...")
    # if(compare_element == 'titulo'):
    #     print("Comparar por titulo")
    #     test = source_file_content['Title'].tolist()

    # elif(compare_element == 'contenido'):
    #     print("Comparar por contenido")
    #     test = source_file_content['Content'].tolist()
    # else:
    #     print("Comparar por titulo y contenido")
    #     titletest = source_file_content['Title'].tolist()
    #     contenttest = source_file_content['Content'].tolist()
    #     test = [itemT + " " + (itemC if itemC is not None else "") for itemT, itemC in
    #                    zip_longest(titletest, contenttest, fillvalue="")]


    # Unigramas
    print("Determinando unigramas/bigramas...")
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
        print("Vectorizando...")
    except KeyError:
        raise ValueError(f'Tipo de vectorización o característica no reconocidos: {vector_type}, {feature_type}')
    
    Y = vectorizer.fit_transform(test)

    print (type(Y.toarray))

    # Guardar vectorización de test
    print("Guardando vectorización de test...")
    output_folder = os.path.join(os.getcwd(), 'vectorized_test')    # si se prueba directamente, el folder se crea en el workning dir, y no en backend
    os.makedirs(output_folder, exist_ok=True)
    filename = f'test_{compare_element}_{vector_type}_{feature_type}.pkl'
    filepath = os.path.join(output_folder, filename)
    with open(filepath, 'wb') as file:
        pickle.dump(Y, file)
    print(f'Vectorización de test guardado: {filename}')


if __name__ == "__main__":
    main()


# QUIZÁS HAGA FALTA IMPEMENTAR QUE SE BORREN LOS CONTENIDOS DE LA CARPETA DE vectorized_test PARA NO DAR ERRORES
# A LA HORA DE APLICAR LA SIMILITUD COSENO (SELECCIONAR UNA ENTRE MUCHAS VECTORIZACIONES ANTERIORES)