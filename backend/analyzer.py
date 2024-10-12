from vectorization import *

def analyzer(vector_type, feature_type, compare_element, source_file):
    # Recuperar el corpus con base en el elemento a comparar

    if(compare_element == 'titulo'):
        print("Comparar por titulo")
        corpus = source_file['Title'].tolist()

    elif(compare_element == 'contenido'):
        print("Comparar por contenido")
        corpus = source_file['Content'].tolist()
    else:
        print("Comparar por titulo y contenido")
        titleCorpus = source_file['Title'].tolist()
        contentCorpus = source_file['Content'].tolist()
        corpus = [itemT + " " + (itemC if itemC is not None else "") for itemT, itemC in
                       zip_longest(titleCorpus, contentCorpus, fillvalue="")]

    # Escoger la configuración con base a los parametros
    if(feature_type == 'unigramas'):
        if(vector_type == 'frecuencia'):
            print("Frecuencia de unigramas")
            x = freqUniVectorize(corpus)

        elif(vector_type == 'binarizado'):
            print("Binario de unigramas")
            x = oneHotUniVectorize(corpus)

        elif(vector_type == 'tfidf'):
            print("TF-IDF de unigramas")
            x = tfidfUniVectorize(corpus)
    else:
        if(vector_type == 'frecuencia'):
            print("Frecuencia de bigramas")
            x = freqBiVectorize(corpus)

        elif(vector_type == 'binarizado'):
            print("Binario de bigramas")
            x = oneHotBiVectorize(corpus)

        elif(vector_type == 'tfidf'):
            print("TF-IDF de bigramas")
            x = tfidfBiVectorize(corpus)

    return x
