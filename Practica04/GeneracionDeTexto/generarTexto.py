import pandas as pd
import os.path
import random

def loadModel(model_filename):
    diretory = os.path.join(os.path.dirname(os.getcwd()),'ModelosDeLenguage\\LanguageModels')
    try:   
        filepah = os.path.join(diretory,model_filename)
        model = pd.read_csv(filepah,sep='\t')
        return model
    except Exception as e:
        print(f'An exception occurred: {e}')
        return e
    
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


def generar_texto(model_filename, feature, ngramStart):
    model = loadModel(model_filename)
    if(feature == 'bi'):
        generatedText = bigramTextGeneration(model,ngramStart)
    else:
        generatedText = trigramTextGeneration(model,ngramStart)
    return generatedText


def main():
    feature = 'tri'
    model_filename = 'trigram_language_model_adair.tsv'
    ngramStart = 'creo que'

    model = loadModel(model_filename)

    if(feature == 'bi'):
         generatedText = bigramTextGeneration(model,ngramStart)
    else:
        generatedText = trigramTextGeneration(model,ngramStart)

    print(generatedText)
   

if __name__ == "__main__":
    main()