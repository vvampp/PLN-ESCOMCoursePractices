import pandas as pd
import os.path

def loadModel(model_filename):
    # diretory = os.path.join(os.path.dirname(os.getcwd()),'ModelosDeLenguage\\LanguageModels')

    # Para ejecutar en el pycharm alv
    print(os.getcwd())
    diretory = os.path.join(os.path.dirname(os.getcwd()),'Backend/ModelosDeLenguage/LanguageModels')
    try:   
        filepah = os.path.join(diretory,model_filename)
        model = pd.read_csv(filepah,sep='\t')
        return model
    except Exception as e:
        print(f'An exception occurred: {e}')
        return e
    
def searchBigram(model,ngramStart):
    w = ngramStart.split()
    if(len(w)!=1):
        raise Exception ("Cantidad de argumentos no valida")
    appearances = model.index[model['Term 1'].str.strip() == w[0]].tolist()
    searchResult = appearances[:3]
    return searchResult

def searchTrigram(model,ngramStart):
    w = ngramStart.split()
    if(len(w)!=2):
        raise Exception ("Cantidad de argumentos no valida")
    appearances = model.index[(model['Term 1'].str.strip() == w[0]) & (model['Term 2'].str.strip() == w[1])].tolist()
    searchResult = appearances[:3]
    return searchResult, w[1]

def bigramPrediction(model,ngramStart):
    predictedText = ngramStart
    while ngramStart != '.':
        indexes = searchBigram(model,ngramStart)
        nextWord = []
        for index in indexes:
            if(model['Term 2'].iloc[index] != '#'):
                nextWord.append(model['Term 2'].iloc[index])
        nextWord.append(".")
        print(nextWord)
        choice = int(input())
        predictedText = predictedText + " " + nextWord[choice]
        ngramStart = nextWord[choice]
    return predictedText

def trigramPrediction(model,ngramStart):
    predictedText = ngramStart
    print (ngramStart)
    while ngramStart.split()[1] != '.':
        indexes,nextT1 = searchTrigram(model,ngramStart)
        nextWord = []
        for index in indexes:
            if(model['Term 3'].iloc[index] != '#'):
                nextWord.append(model['Term 3'].iloc[index])
        nextWord.append(".")
        print(nextWord)
        choice = int(input())
        predictedText = predictedText + " " + nextWord[choice]
        ngramStart = nextT1 + " " + nextWord[choice]
    return predictedText


def most_probable_words_bigram(model, ngramStart):
    while ngramStart != '.':
        indexes = searchBigram(model, ngramStart)
        nextWord = []
        for index in indexes:
            if (model['Term 2'].iloc[index] != '#'):
                nextWord.append(model['Term 2'].iloc[index])
        nextWord.append(".")
        # Enviar a front end
        return nextWord


def most_probable_words_trigram(model, ngramStart):
    print(ngramStart)
    while ngramStart.split()[1] != '.':
        indexes, nextT1 = searchTrigram(model, ngramStart)
        nextWord = []
        for index in indexes:
            if (model['Term 3'].iloc[index] != '#'):
                nextWord.append(model['Term 3'].iloc[index])
        nextWord.append(".")
        return nextWord

def get_palabras_probables(model_filename, feature, ngramStart):
    model = loadModel(model_filename)
    if(feature == 'bi'):
        most_probable_words = most_probable_words_bigram(model,ngramStart)
    else:
        most_probable_words = most_probable_words_trigram(model,ngramStart)
    return most_probable_words


def main ():
    # Recibido desde front end 
    feature = 'tri'               # bi / tri
    model_filename = 'trigram_language_model_adair.tsv'
    ngramStart = 'material del'

    model = loadModel(model_filename)

    if(feature == 'bi'):
        predictedText = bigramPrediction(model,ngramStart)
    else:
        predictedText = trigramPrediction(model,ngramStart)

    print(predictedText)

if __name__ == "__main__":
    main()