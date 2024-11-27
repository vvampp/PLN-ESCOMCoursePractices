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

from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import LabelEncoder

from scipy.sparse import csr_matrix
from scipy.sparse import vstack


nlp = spacy.load('es_core_news_sm')
spell = SpellChecker(language='es')


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

    return data_set


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


def lemmatizationStopWords(features):
    doc = nlp(features)
    tokens = [token.lemma_.lower() for token in doc if not token.pos_ in ["DET", "ADP", "CCONJ", "SCONJ", "PRON"]]
    return " ".join(tokens)


def normalize(data):
    normalized_features = data['features'].apply(lemmatizationStopWords)
    normalized_data = pd.DataFrame({
        'features': normalized_features,
        'target' : data['target']
    })
    print('Normalizacion completada')
    return normalized_data


def repeatedWords(data):
    data['features'] = data['features'].apply(removeRepeatedWords)
    print('repeated words completado')
    return data


def negations(data):
    data['features'] = data['features'].apply(handleNegations)
    print('negation handling completado')
    return data

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
        print(f" Error al cargar el lexicon SEL: {e}")
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


def vectorizeData(data, lexicon_sel, vectorizador, is_training=True):
    try:
        if is_training:
            # Ajusta el vectorizador y transforma las características
            vector_matrix = vectorizador.fit_transform(data['features'])
        else:
            # Usa el vectorizador ya ajustado para transformar las características
            vector_matrix = vectorizador.transform(data['features'])
        
        print(f"Matrix Shape: {vector_matrix.shape}")
    except Exception as e:
        print(f"Error al generar la matriz : {e}")
        raise

    # Convertir 'features' a lista de cadenas
    cadenas = data['features'].tolist()

    # Agregar características de SEL
    try:
        final_matrix = add_sentiment_features(vector_matrix, cadenas, lexicon_sel)
    except Exception as e:
        print(f"Error al agregar características de SEL: {e}")
        raise

    print(f"Matriz final procesada con éxito: {final_matrix.shape}")
    return final_matrix



def applySMOTE(X_train, y_train, target_classes=[1, 2, 3],target_samples=5000, random_state=0):
        # Convertir y_train a numpy array si no lo es
    if isinstance(y_train, pd.Series):
        y_train = y_train.values

    if not isinstance(X_train, csr_matrix):
        X_train = csr_matrix(X_train)
    
    # Crear una copia para modificar solo las clases objetivo
    X_filtered = []
    y_filtered = []

    for cls in target_classes:
        indices = np.where(y_train == cls)[0]
        X_filtered.append(X_train[indices])
        y_filtered.extend(y_train[indices])

    # Concatenar los datos filtrados
    X_filtered = vstack(X_filtered) if len(X_filtered) > 1 else X_filtered[0]
    y_filtered = np.array(y_filtered)

    # Convertir a formato denso para SMOTE
    X_filtered_dense = X_filtered.toarray()
    
    # Definir la estrategia de muestreo personalizada
    sampling_strategy = {cls: target_samples for cls in target_classes}

    # Aplicar SMOTE
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_smote_dense, y_smote = smote.fit_resample(X_filtered_dense, y_filtered)

    # Convertir datos SMOTE de nuevo a formato disperso
    X_smote = csr_matrix(X_smote_dense)

    # Combinar las clases SMOTE con las originales no seleccionadas
    other_classes = [cls for cls in np.unique(y_train) if cls not in target_classes]
    X_other = []
    y_other = []

    for cls in other_classes:
        indices = np.where(y_train == cls)[0]
        X_other.append(X_train[indices])
        y_other.extend(y_train[indices])

    if X_other:
        X_other = vstack(X_other) if len(X_other) > 1 else X_other[0]
        X_smote_combined = vstack([X_other, X_smote])
        y_smote_combined = np.hstack([y_other, y_smote])
    else:
        X_smote_combined = X_smote
        y_smote_combined = y_smote

    with open('X_combined_SMOTE.pkl', 'wb') as pkl_file:
        pickle.dump(X_smote_combined, pkl_file)
        print('Archivo guardado: X_combined_SMOTE.pkl')

    with open('y_combined_SMOTE.pkl', 'wb') as pkl_file:
        pickle.dump(y_smote_combined, pkl_file)
        print('Archivo guardado: y_combined_SMOTE.pkl')

    return X_smote_combined, y_smote_combined



def balanceTrainingSet(X_train, y_train):
    # Convertir X_train a csr_matrix si no lo es
    if not isinstance(X_train, csr_matrix):
        X_train = csr_matrix(X_train)

    # Convertir y_train a un array numpy si no lo es
    if isinstance(y_train, pd.Series):
        y_train = y_train.values
    
    # Verificación de la distribución de clases antes de balancear
    print("Distribución de clases antes del balanceo:")
    print(pd.Series(y_train).value_counts())  # Convertir a pandas.Series temporalmente
    
    # Convertir y_train a un array numpy para facilitar la manipulación
    y_train = np.array(y_train)
    
    # Tamaño de la clase 3 (referencia) y cálculo del promedio por clase
    class_3_size = 2 *np.sum(y_train == 3) 
    target_sizes = {
        label: int((np.sum(y_train == label) + class_3_size) / 2)
        for label in np.unique(y_train)
    }
    
    # Inicializar matrices para características y etiquetas balanceadas
    balanced_X = None
    balanced_y = []

    for label, target_size in target_sizes.items():
        # Filtrar los índices de la clase actual
        indices = np.where(y_train == label)[0]
        num_samples = len(indices)
        
        if num_samples > target_size:  # Undersampling
            sampled_indices = np.random.choice(indices, size=target_size, replace=False)
        else:  # Oversampling
            sampled_indices = np.random.choice(indices, size=target_size, replace=True)
        
        # Extraer las filas correspondientes de X_train y las etiquetas
        sampled_X = X_train[sampled_indices, :]
        sampled_y = y_train[sampled_indices]

        # Agregar a las matrices balanceadas
        if balanced_X is None:
            balanced_X = sampled_X
        else:
            balanced_X = vstack([balanced_X, sampled_X])  # Concatenar matrices dispersas

        balanced_y.extend(sampled_y)
    
    # Convertir balanced_y de nuevo a un array
    balanced_y = np.array(balanced_y)
    
    # Verificación de la distribución de clases después del balanceo
    print("\nDistribución de clases después del balanceo:")
    print(np.unique(balanced_y, return_counts=True))
    
    return balanced_X, balanced_y




def testClassifiers(X_train, y_train):
    if not isinstance(X_train, csr_matrix):
        X_train = csr_matrix(X_train)

    if not isinstance(y_train, pd.Series):
        y_train = pd.Series(y_train)

    y_train = y_train.reset_index(drop=True)

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
        print(f"{name} - Mejores parámetros: {grid.best_params_}")

        fold_reports = []
        scores = []

        for train_idx, val_idx in skf.split(X_train, y_train):
            X_fold_train, X_fold_val = X_train[train_idx], X_train[val_idx]
            y_fold_train, y_fold_val = y_train.iloc[train_idx], y_train.iloc[val_idx]  # Uso de iloc para indexar y_train

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
    # Corpus a DataSet
    data_set = createDataSet()

    # Normalización del DataSet
    if os.path.exists('normalized_data_set.tsv'):          
        normalized_data = pd.read_csv('normalized_data_set.tsv', sep='\t')
        print("Archivo guardado: normalized_data_set.tsv")    
    else:
        normalized_data = repeatedWords(data_set)
        normalized_data = normalize(normalized_data)
        normalized_data = negations(normalized_data)
        normalized_data.to_csv('normalized_data_set.tsv',sep='\t', index=False)

    # División del Data Set
    train_data, test_data = train_test_split(normalized_data, test_size=0.2, stratify=normalized_data["target"], random_state=0, shuffle=True)

    # TF - IDF y  Lexicon Features
    if os.path.exists('lexicon_sel.pkl'):
         with open('lexicon_sel.pkl', 'rb') as lexicon_file:
            lexicon_sel = pickle.load(lexicon_file)
    else:
        lexicon_sel = load_sel()
        with open('lexicon_sel.pkl', 'wb') as lexicon_file:
            pickle.dump(lexicon_sel, lexicon_file)

    vectorizador = TfidfVectorizer()
    feature_train_matrix = vectorizeData(train_data, lexicon_sel, vectorizador, is_training=True)
    feature_test_matrix = vectorizeData(test_data,lexicon_sel, vectorizador, is_training=False)

    # Balanceo de feature_train_matrix junto con train_data['target']
    #balanced_features, balanced_targets = applySMOTE(feature_train_matrix, train_data['target'], target_classes=[1,2,3])
    balanced_features, balanced_targets = balanceTrainingSet(feature_train_matrix,train_data['target'])

    # cross validation con opciones de clasificadores
    # testClassifiers(balanced_features,balanced_targets)

    # entrenamiento del modelo
    X_train = balanced_features
    y_train = balanced_targets
    X_test = feature_test_matrix
    y_test = test_data['target']

    svc = SVC(C=10, kernel='rbf', class_weight='balanced', random_state=0)
    print("Entrenando modelo  SVC - {'C': 10, 'kernel': 'rbf', class_weight: 'balanced'} ...")
    svc.fit(X_train,y_train)

    with open('SVC_Model.pkl', 'wb') as pkl_file:
        pickle.dump(svc,pkl_file)
    print("Modelo SVC guardado exitosamente en 'SVC_Model.pkl'.")

    # test del modelo
    print("Evaluando modelo clasificador")
    y_pred = svc.predict(X_test)
    print(classification_report(y_test,y_pred))


if __name__ == '__main__':
    main()