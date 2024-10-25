import pandas as pd
import os.path
import random

def loadModel(model_filename):
    directory = os.path.join(os.path.dirname(os.getcwd()),'Backend', 'ModelosDeLenguage', 'LanguageModels')
    file_path = os.path.join(directory, model_filename)
    
    try:
        model = pd.read_csv(file_path, sep='\t')
        return model
    except FileNotFoundError:
        print(f'El archivo no fue encontrado: {file_path}')
        return None 
    except Exception as e:
        print(f'Ocurrió un error: {e}')
        return None 
    
def searchBigram(model,bigramStart):
    w = bigramStart.split()
    if(len(w)!=1):
        raise Exception ("Cantidad de argumentos no valida.")
    apperances = model.index[model['Term 1'].str.strip() == w[0]].tolist()
    return apperances

def searchTrigram(model,trirgamStart):
    w = trirgamStart.split()
    if(len(w)!=2):
        raise Exception ("Cantidad de argumentos no valida")
    appearances = model.index[(model['Term 1'].str.strip() == w[0]) & (model['Term 2'].str.strip() == w[1])].tolist()
    return appearances, w[1]


def bigramTextGeneration(model,bigramStart):
    generatedText = bigramStart
    probabilities = []
    while bigramStart != '#':
        indexes = searchBigram(model,bigramStart)
        if not indexes:
            break

        probabilities= [model['Conditional Probability of Bigram'].iloc[index] for index in indexes]

        total_prob = sum(probabilities)
        normalized_probs = [prob / total_prob for prob in probabilities]
        acc_probs = []
        acc_sum = 0

        for prob in normalized_probs:
            acc_sum += prob
            acc_probs.append(acc_sum)

        rand_val = random.random()
        selected_index = None

        for i, accProb in enumerate(acc_probs):
            if rand_val <= accProb:
                selected_index = indexes[i]
                break

        bigramStart = model['Term 2'].iloc[selected_index]
        generatedText += " " + bigramStart
    return generatedText
        

def trigramTextGeneration(model,trirgamStart):
    generatedText = trirgamStart
    probabilities = []
    while trirgamStart != '#':
        indexes, nextT1 = searchTrigram(model,trirgamStart)
        if not indexes:
            break

        probabilities= [model['Conditional Probability of Trigram'].iloc[index] for index in indexes]

        total_prob = sum(probabilities)
        normalized_probs = [prob / total_prob for prob in probabilities]
        acc_probs = []
        acc_sum = 0

        for prob in normalized_probs:
            acc_sum += prob
            acc_probs.append(acc_sum)

        rand_val = random.random()
        selected_index = None

        for i, accProb in enumerate(acc_probs):
            if rand_val <= accProb:
                selected_index = indexes[i]
                break
        
        trirgamStart = nextT1 + " " + model['Term 3'].iloc[selected_index]
        generatedText += " " + model['Term 3'].iloc[selected_index]
    return generatedText

def getNext(model,start):
    probabilities = []
    
    indexes = searchBigram(model,start)

    probabilities= [model['Conditional Probability of Trigram'].iloc[index] for index in indexes]

    total_prob = sum(probabilities)
    normalized_probs = [prob / total_prob for prob in probabilities]
    acc_probs = []
    acc_sum = 0

    for prob in normalized_probs:
        acc_sum += prob
        acc_probs.append(acc_sum)

    rand_val = random.random()
    selected_index = None

    for i, accProb in enumerate(acc_probs):
        if rand_val <= accProb:
            selected_index = indexes[i]
            break

    next = model['Term 2'].iloc[selected_index]
    return next

def generar_texto(model_filename, feature, ngramStart):
    model = loadModel(model_filename)
    
    if model is None:
        raise Exception(f"No se pudo cargar el modelo desde {model_filename}. Verifica la ruta y el formato del archivo.")
    
    if feature == 'bi':
        generatedText = bigramTextGeneration(model, ngramStart)
    else:
        ngramStart = ngramStart + " " + getNext(model, ngramStart)
        generatedText = trigramTextGeneration(model, ngramStart)

    return generatedText
