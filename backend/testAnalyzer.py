import os.path
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
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
        

def main():
    test_txt_file = 'test.txt'

    try:
        with open(test_txt_file, 'r') as file:
            test_file_content = file.read().rstrip()
    except Exception as e:
        print(f"An exception ocurred: {e}")

    normalized_test = normalizeTest(test_file_content)
    normalized_test = [normalized_test]
    print(normalized_test)


    # Parámetros
    compare_element = 'tyc'  # 'title', 'content', 'tyc'
    vector_type = 'tfidf'    # 'freq', 'onehot', 'tfidf'
    feature_type = 'uni'     # 'uni', 'bi'

    vectorizer = load_vectorizer(compare_element, vector_type, feature_type)

    
    print(vectorizer)

    vectorized_test = vectorizer.transform(normalized_test)

    print(vectorized_test)
    

if __name__ == "__main__":
    main()