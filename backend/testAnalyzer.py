import os.path
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import spacy


def normalizeTest(test_file_content):

    nlp = spacy.load('es_core_news_sm')

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
            print(filename)
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
            print(filename)
    else:
        raise FileNotFoundError(f'La matriz de corpus buscada no existe: {filename}')
    
    return vectorized_corpus


def cosineSimilarity (test_vector, corpus_matrix):
    cosine_similarities = cosine_similarity(test_vector,corpus_matrix)

    cosine_similarities = cosine_similarities[0] 

    top_10_index = np.argsort(cosine_similarities)[-10:][::-1]
    
    return top_10_index, cosine_similarities[top_10_index]



def testAnalyzer(test_txt_file,compare_element,vector_type,feature_type):
    # test_txt_file = 'test.txt'
    # print(compare_element,vector_type,feature_type)

    try:
        with open(test_txt_file, 'r') as file:
            test_file_content = file.read().rstrip()
    except Exception as e:
        print(f"An exception ocurred: {e}")

    normalized_test = normalizeTest(test_file_content)
    normalized_test = [normalized_test]

    #print(normalized_test)

    # Parámetros
    # compare_element = 'content'  # 'title', 'content', 'tyc'
    # vector_type = 'onehot'    # 'freq', 'onehot', 'tfidf'
    # feature_type = 'uni'     # 'uni', 'bi'


    vectorizer = load_vectorizer(compare_element, vector_type, feature_type) 
    vectorized_test = vectorizer.transform(normalized_test)
    #print(vectorized_test)

    corpus_matrix = load_vectorized_corpus(compare_element,vector_type,feature_type)


    top_10_index, top_10_similarities = cosineSimilarity(vectorized_test,corpus_matrix)

    document_similarity = []
    i = 1
    # para debbuging
    for idx, similarity in zip (top_10_index, top_10_similarities):
        print(f'Documento {idx} - Similitud: {similarity:.5f}')
        document_similarity.append((f'{i}: Indice de documento: {idx}', f'Similitud: {similarity:.5f}'))
        i += 1

    return document_similarity