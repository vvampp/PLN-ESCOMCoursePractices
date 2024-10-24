import os.path
import pandas as pd


def loadModels(feature):
    diretory = os.path.join(os.path.dirname(os.getcwd()),'ModelosDeLenguage\\LanguageModels')
    models = []
    if(feature=='bi'):
        model_filenames = ['bigram_language_model_adair.tsv','bigram_language_model_bambino.tsv','bigram_language_model_Rojo.tsv']
    else:
        model_filenames = ['trigram_language_model_adair.tsv','trigram_language_model_bambino.tsv','trigram_language_model_Rojo.tsv'] 

    for filename in model_filenames:
            try:   
                filepah = os.path.join(diretory,filename)
                models.append(pd.read_csv(filepah,sep='\t'))
            except Exception as e:
                print(f'An exception occurred: {e}')
    return models, model_filenames
            

def getBigramProbabilities(w1,w2,models):
    bigramProbability = []

    for model in models:
        context_freq = 0
        bigram_frequency = 0
        vocabulary_size = model['Term 1'].nunique()

        context_rows = model[model['Term 1'] == w1]
        if not context_rows.empty:
            context_freq = context_rows['Frequency of context'].iloc[0]

            bigram_row = context_rows[context_rows['Term 2'] == w2]
            if not bigram_row.empty:
                bigram_frequency = bigram_row['Frequency of Bigram'].iloc[0]
                bigramProbability.append((bigram_frequency+1)/(context_freq + vocabulary_size))
            else:
                bigramProbability.append(1 / (context_freq + vocabulary_size))
        
        else:
            bigramProbability.append(1/vocabulary_size)
        print(f"Vocabulary Size: {vocabulary_size}, W1: {w1}, W2: {w2}, Bigram Frequency: {bigram_frequency}, Context Frequency: {context_freq}, Probability: {bigramProbability[-1]}")
    return bigramProbability


def getTrigramProbabilities(w1,w2,w3,models):
    trigramProbability = []

    for model in models:
        bigram_count = model[['Term 1','Term 2']].drop_duplicates().shape[0]

        context_rows_2 = model[(model['Term 1'] == w1) & (model['Term 2'] == w2)]
        if not context_rows_2.empty:
            context_freq_2 = context_rows_2['Frequency of Context'].iloc[0]

            trigram_row = context_rows_2[context_rows_2['Term 3'] == w3]
            if not trigram_row.empty:
                trigram_freq = trigram_row['Frequency of Trigram'].iloc[0]
                trigramProbability.append((trigram_freq+1)/(context_freq_2+bigram_count))
            
            else:
                trigramProbability.append(1/(context_freq_2+bigram_count))
        else:
            trigramProbability.append(1/bigram_count)
    print(trigramProbability)
    return trigramProbability


def main():
    feature = 'bi'
    models,model_filenames = loadModels(feature)

    test_sentence = 'hola hola'
    test_sentence =  "$ " + test_sentence + " #"
    words = test_sentence.split()

    sentenceProbability = [1,1,1]

    if(feature == 'bi'):
        for i in range(len(words)-1):
            probabilities = getBigramProbabilities(words[i],words[i+1],models)
            sentenceProbability = [x * y for x, y in zip(sentenceProbability, probabilities)]
            print('\n')
    else:
        for i in range(len(words)-2):
            probabilities = getTrigramProbabilities(words[i],words[i+1],words[i+2],models)
            sentenceProbability = [x * y for x, y in zip(sentenceProbability, probabilities)]
            print('\n')

    formatedProbabilities = [(model_filenames[i],probability) for i,probability in enumerate(sentenceProbability)]
    sortedProbabilities = sorted(formatedProbabilities, key=lambda x:x[1],reverse=True)

    print(sortedProbabilities)

    

if __name__ == "__main__":
    main()