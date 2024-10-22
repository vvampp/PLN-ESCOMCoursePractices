import os.path
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import spacy

nlp = spacy.load('es_core_news_sm')


def normalizeTest(test_file_content):

    content = nlp(test_file_content)
    nContent = ""

    for token in content:
        if not token.pos_ in ["DET","ADP", "CCONJ", "SCONJ", "PRON"]:
            nContent = nContent + token.lemma_ + " "

    return nContent



def load_vectorizer(compare_element, vector_type, feature_type):
    filename = f'{compare_element}_{vector_type}_{feature_type}_vectorizer.pkl'

    vectorizer_folder = os.path.join(os.getcwd(),'vectorizers')
    vectorizer_filepath = os.path.join(vectorizer_folder,filename)
    if os.path.exists(vectorizer_filepath):
        with open(vectorizer_filepath, 'rb') as file:
            vectorizer = pickle.load(file)
    else:
        raise FileNotFoundError(f'El archivo del vectorizador buscado no existe: {vectorizer_filepath}')
    
    return vectorizer



def load_vectorized_corpus(compare_element, vector_type, feature_type):
    filename = f'{compare_element}_{vector_type}_{feature_type}.pkl'

    vectorized_corpus_folder = os.path.join(os.getcwd(), 'vectorized_data')
    vectorized_corups_filepath = os.path.join(vectorized_corpus_folder,filename)
    if os.path.exists(vectorized_corups_filepath):
        with open(vectorized_corups_filepath, 'rb') as file:
            vectorized_corpus = pickle.load(file)
    else:
        raise FileNotFoundError(f'La matriz de corpus buscada no existe: {filename}')
    
    return vectorized_corpus


def cosineSimilarity (test_vector, corpus_matrix):
    cosine_similarities = cosine_similarity(test_vector,corpus_matrix)

    cosine_similarities = cosine_similarities[0] 

    return cosine_similarities



def main():
    test_txt_file1 = "Apple_titulo-contenido.txt"
    test_txt_file2 = "CFE_titulo-contenido_nuevo.txt"
    test_txt_file3 = "Liguilla_nuevo.txt"
    test_txt_file4 = "Papa_titulo.txt"
    test_txt_file5 = "Trump_contenido.txt"

    tests_txt = [test_txt_file1, test_txt_file2, test_txt_file3, test_txt_file4, test_txt_file5]
    normalized_tests = []

    # Normaliza los tests
    for test_txt in tests_txt:
        try:
            with open(test_txt, 'r') as file:
                content = file.read().rstrip()
                normalized = normalizeTest(content)
                normalized_tests.append([normalized])
        except Exception as e:
            print(f"An exception occurred: {e}")

    compare_elements = ['title', 'content', 'tyc']
    vector_types = ['freq', 'onehot', 'tfidf']
    feature_types = ['uni', 'bi']

    # Para cada test, almacenar las 10 mayores similitudes y sus parámetros
    top_similarities_tests = []

    for normalized_test in normalized_tests:
        cosine_similarities_with_params = []  # Lista para todas las similitudes coseno del test, con parámetros

        # Realiza comparaciones por cada combinación de elementos
        for compare_element in compare_elements:
            for vector_type in vector_types:
                for feature_type in feature_types:
                    # Carga la matriz del corpus y vectoriza el test
                    corpus_matrix = load_vectorized_corpus(compare_element, vector_type, feature_type)
                    vectorizer = load_vectorizer(compare_element, vector_type, feature_type)
                    vectorized_test = vectorizer.transform(normalized_test)

                    # Calcula similitud coseno
                    cos_sim = cosineSimilarity(vectorized_test, corpus_matrix)

                    # Añade las similitudes junto con los parámetros correspondientes
                    for idx,sim in enumerate(cos_sim.flatten()):
                        cosine_similarities_with_params.append({
                            'similarity': sim,
                            'compare_element': compare_element,
                            'vector_type': vector_type,
                            'feature_type': feature_type,
                            'corpus_index': idx
                        })

        # Ordena las similitudes en función de 'similarity' y selecciona las 10 mayores
        sorted_similarities = sorted(cosine_similarities_with_params, key=lambda x: x['similarity'], reverse=True)
        top_10_similarities = sorted_similarities[:10]

        # Almacena los resultados del test actual
        top_similarities_tests.append(top_10_similarities)

    # Muestra las 10 mayores similitudes para cada test junto con sus parámetros
    for i, top_similarities in enumerate(top_similarities_tests):
        print(f"Top 10 similitudes del test {i+1}:")
        for entry in top_similarities:
            print(f"Corpus document: {entry['corpus_index']}\t Vector representation: {entry['vector_type']}\t "f"Extracted features: {entry['feature_type']}\t Comparison element: {entry['compare_element']} \t Similarity: {entry['similarity']}")



if __name__ == "__main__":
    main()