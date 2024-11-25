import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import spacy
from spellchecker import SpellChecker
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
import numpy as np
import pickle
import os

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, cross_validate, StratifiedKFold
from sklearn.metrics import classification_report, make_scorer, f1_score

from scipy.sparse import csr_matrix

nlp = spacy.load('es_core_news_sm')
spell = SpellChecker(language='es')

def load_sel():
    lexicon_sel = {}
    try:
        with open('SEL_full.txt', 'r') as input_file:
            for line in input_file:
                palabras = line.split("\t")
                palabra = palabras[0]
                palabras[6]= re.sub('\n', '', palabras[6])
                emocion = palabras[6].strip()
                valor = palabras[5].strip()

                # Agregar al diccionario
                pair = (emocion, valor)
                if palabra not in lexicon_sel:
                    lexicon_sel[palabra] = [pair]
                else:
                    lexicon_sel[palabra].append(pair)

        if 'palabra' in lexicon_sel:
            del lexicon_sel['palabra']  
    except Exception as e:
        print(f"[ERROR] Error al cargar el lexicon SEL: {e}")
        raise



def getSELFeatures(cadenas, lexicon_sel):

    emocion_a_clave = {
        'Alegría': '__alegria__',
        'Tristeza': '__tristeza__',
        'Enojo': '__enojo__',
        'Repulsión': '__repulsion__',
        'Miedo': '__miedo__',
        'Sorpresa': '__sorpresa__'
    }
  
    features = []
    cadenas_con_valores = []  

    for i, cadena in enumerate(cadenas):
        valores = {clave: 0.0 for clave in emocion_a_clave.values()}

        # Normalizar palabras en la cadena
        cadena_palabras = re.split(r'\s+', cadena)


        # Calcular valores SEL
        for palabra in cadena_palabras:
            if palabra in lexicon_sel:
                for emocion, valor in lexicon_sel[palabra]:
                    clave = emocion_a_clave.get(emocion)
                    if clave:  # Si la emoción está en el mapeo
                        valores[clave] += float(valor)
                        

        valores['acumuladopositivo'] = valores['__alegria__'] + valores['__sorpresa__']
        valores['acumuladonegative'] = valores['__enojo__'] + valores['__miedo__'] + valores['__repulsion__'] + valores['__tristeza__']
        #print(f'Acumulado positivo: ${valores['acumuladopositivo']} \t Acumulado negativo: ${valores['acumuladonegative']}')

        # Registrar cadenas con valores
        if valores['acumuladopositivo'] > 0 or valores['acumuladonegative'] > 0:
            cadenas_con_valores.append((cadena, valores['acumuladopositivo'], valores['acumuladonegative']))

        features.append(valores)

    return features




def add_sentiment_features(matrix, cadenas, lexicon_sel):

    polaridad = getSELFeatures(cadenas, lexicon_sel)

    polaridad_pos = np.array([p['acumuladopositivo'] for p in polaridad]).reshape(-1, 1)
    polaridad_neg = np.array([p['acumuladonegative'] for p in polaridad]).reshape(-1, 1)
    final_matrix = hstack([matrix, polaridad_pos, polaridad_neg])
    try:
        final_matrix = hstack([matrix, polaridad_pos, polaridad_neg])
    except Exception as e:
        print(f"Error al combinar matrices: {e}")
        raise

    print(f"Matriz combinada creada con dimensiones: {final_matrix.shape}")
    return final_matrix


def postNormalizationWithTFIDF(data, lexicon_sel):
    vectorizador = TfidfVectorizer()
    try:
        tfidf_matrix = vectorizador.fit_transform(data['features'])
        print(f"TF-IDF Matrix Shape: {tfidf_matrix.shape}")
    except Exception as e:
        print(f"Error al generar la matriz TF-IDF: {e}")
        raise

    # Convertir 'features' a lista de cadenas
    cadenas = data['features'].tolist()

    # Agregar características de SEL
    try:
        final_matrix = add_sentiment_features(tfidf_matrix, cadenas, lexicon_sel)
    except Exception as e:
        print(f"Error al agregar características de SEL: {e}")
        raise

    print(f"Matriz final procesada con éxito: {final_matrix.shape}")
    return final_matrix


def postNormalizationWithFrequency(data,lexicon_sel):
    try:
        vectorizador = CountVectorizer()
        frequency_matrix = vectorizador.fit_transform(data['features'])
        print(f"Frequency Matrix Shape: {frequency_matrix.shape}")
    except Exception as e:
        print(f"Error al generar la matriz de Frecuencia: {e}")
        raise  

    cadenas = data['features'].tolist()

    try: 
        final_matrix = add_sentiment_features(frequency_matrix, cadenas, lexicon_sel)
    except Exception as e:
        print(f"Error al agregar características de SEL: {e}")
        raise

    print(f"Matriz final procesada con éxito: {final_matrix.shape}")
    return final_matrix
    

def createDataSet():
    try:
        df = pd.read_excel('Rest_Mex_2022.xlsx', header=None)
    except Exception as e:
        print(f"An exception occurred: {e}")
        return
    df = df.iloc[1:]

    df[0] = df[0].astype(str).str.replace(r'[\r\n]', ' ', regex=True)
    df[1] = df[1].astype(str).str.replace(r'[\r\n]', ' ', regex=True)
    
    df['features'] = df[0].astype(str) + " " + df[1].astype(str)
    df['target'] = df[2]
    data_set = df[['features','target']]

    print("Distribución original de clases:")
    print(data_set["target"].value_counts())
    print("Porcentaje antes del balanceo:")
    print(df["target"].value_counts(normalize=True) * 100)

    train_data, test_data = train_test_split(data_set, test_size=0.2, stratify=df["target"], random_state=0, shuffle=True)

    return train_data, test_data


def balanceClasses(train_data):
    train_data_augmented = train_data.copy()
    
    class_counts = train_data_augmented["target"].value_counts()

    oversampling_factors = {
        1: 2.2,  
        2: 1.8,  
        3: 1.5,  
        4: 1.3,  
    }
    undersampling_factor = 0.4  

    balanced_train_data = []

    for polarity, count in class_counts.items():
        class_data = train_data_augmented[train_data_augmented["target"] == polarity]

        if polarity == 5: 
            target_size = int(count * undersampling_factor)
            class_data = resample(class_data, replace=False, n_samples=target_size, random_state=0)
        else:
            factor = oversampling_factors.get(polarity, 1.0)
            target_size = int(count * factor)
            class_data = resample(class_data, replace=True, n_samples=target_size, random_state=0)

        balanced_train_data.append(class_data)

    
    balanced_train_data = pd.concat(balanced_train_data)

    
    print("Distribución después del balanceo:")
    print(balanced_train_data["target"].value_counts())
    print("Porcentaje después del balanceo:")
    print(balanced_train_data["target"].value_counts(normalize=True) * 100)

    return balanced_train_data


def lemmatizationStopWords(features):
    doc = nlp(features)
    tokens = [token.lemma_.lower() for token in doc if not token.pos_ in ["DET", "ADP", "CCONJ", "SCONJ", "PRON"]]
    return " ".join(tokens)

def normalize(balanced_train_data):
    normalized_features = balanced_train_data['features'].apply(lemmatizationStopWords)
    normalized_balanced_data = pd.DataFrame({
        'features': normalized_features,
        'target' : balanced_train_data['target']
    })
    print('Normalizacion completada')
    return normalized_balanced_data


def removeRepeatedWords(input):
    output =  re.sub(r'\b(\w+)( \1\b)+', r'\1', input)
    return re.sub(r'(.)\1{2,}', r'\1\1', output)

def handleNegations(input):
    words = input.split()
    negation_words = {"no", "ni", "nunca", "jamás"}
    negated = False
    result = []
    
    for word in words:
        if negated:
            result.append(f"NO_{word}")
            if word.endswith('.'):
                negated = False
        elif word in negation_words:
            result.append(word)
            negated = True
        else:
            result.append(word)
    return " ".join(result)



def repeatedWords(data):
    data['features'] = data['features'].apply(removeRepeatedWords)
    print('repeated words completado')
    return data

def negations(data):
    data['features'] = data['features'].apply(handleNegations)
    print('negation handling completado')
    return data


def testClassifiers(X_train, y_train):
    models = {
        'LogisticRegression': {
            'model': LogisticRegression(max_iter=1000),
            'params': {
                'C': [0.1, 1, 10]
            }
        },
        'MultinomialNB': {
            'model': MultinomialNB(),
            'params': {
                'alpha': [0.01, 0.1, 1]
            }
        },
        'SVC': {
            'model': SVC(),
            'params': {
                'C': [0.1, 1, 10],
                'kernel': ['linear', 'rbf']
            }
        },
        'MLPC': {
            'model': MLPClassifier(max_iter=1000),
            'params': {
                'hidden_layer_sizes': [(50,), (100,)],
                'activation': ['tanh', 'relu']
            }
        }
    }

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, spec in models.items():
        grid = GridSearchCV(spec['model'], spec['params'], cv=skf, scoring='f1_macro')
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        scores = []
        print(f"{name} - Mejores parámetros: {grid.best_params_}")

        fold_reports = []
        for train_idx, val_idx in skf.split(X_train, y_train):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train[train_idx], y_train[val_idx]

            best_model.fit(X_fold_train, y_fold_train)

            y_pred = best_model.predict(X_fold_val)

            report = classification_report(y_fold_val, y_pred, digits=4)
            fold_reports.append(report)

            fold_f1 = f1_score(y_fold_val, y_pred, average='macro')
            scores.append(fold_f1)

        for i, report in enumerate(fold_reports, 1):
            print(f"[Reporte {i}]\n{report}")

        print(f"{name} - F1 macro por fold: {scores}")
        print(f"{name} - F1 macro promedio: {np.mean(scores)}")






def main():
    try:

        train_data, testing_data = createDataSet()
        balanced_train_data = balanceClasses(train_data)

        if not os.path.exists('final_training_features.pkl'):
            

            if os.path.exists('normalized_train_set.tsv'):          
                normalized_data = pd.read_csv('normalized_train_set.tsv', sep='\t')
         
            else:
                normalized_data = repeatedWords(balanced_train_data)
                normalized_data = normalize(normalized_data)
                normalized_data = negations(normalized_data)
                normalized_data.to_csv('normalized_train_set.tsv',sep='\t', index=False)

            if os.path.exists('lexicon_sel.pkl'):
                with open('lexicon_sel.pkl', 'rb') as lexicon_file:
                    lexicon_sel = pickle.load(lexicon_file)
            else:
                lexicon_sel = load_sel()
                with open('lexicon_sel.pkl', 'wb') as lexicon_file:
                    pickle.dump(lexicon_sel, lexicon_file)
            
            feature_matrix = postNormalizationWithTFIDF(normalized_data, lexicon_sel)
            
            with open('final_training_features.pkl', 'wb') as pkl_file:
                pickle.dump(feature_matrix, pkl_file)
            print("Archivo guardado: final_training_features.pkl")


        with open('final_training_features.pkl', 'rb') as file:
            training_data = pickle.load(file)

        X_train = training_data
        y_train = balanced_train_data['target'].astype(int)

        testClassifiers(X_train,y_train)
        


    except Exception as e:
        print(f'Error durante el flujo principal: {e}')


if __name__ == '__main__':
    main()