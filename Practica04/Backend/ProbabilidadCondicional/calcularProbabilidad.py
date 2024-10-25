import os
import pandas as pd
import spacy

def loadModels(model_filenames):
    directory = os.path.join(os.path.dirname(os.getcwd()),'Backend', 'ModelosDeLenguage', 'LanguageModels')
    models = []

    for filename in model_filenames:
        try:
            filepath = os.path.join(directory, filename)
            models.append(pd.read_csv(filepath, sep='\t'))
        except Exception as e:
            print(f'An exception occurred: {e}')
    return models

def getBigramProbabilities(w1, w2, models):
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
                bigramProbability.append((bigram_frequency + 1) / (context_freq + 1))
            else:
                bigramProbability.append(1 / (context_freq + 1))
        else:
            bigramProbability.append(1 / vocabulary_size)
    print(bigramProbability)
    return bigramProbability

def getTrigramProbabilities(w1, w2, w3, models):
    trigramProbability = []
    for model in models:
        bigram_count = model[['Term 1', 'Term 2']].drop_duplicates().shape[0]

        context_rows_2 = model[(model['Term 1'] == w1) & (model['Term 2'] == w2)]
        if not context_rows_2.empty:
            context_freq_2 = context_rows_2['Frequency of Context'].iloc[0]

            trigram_row = context_rows_2[context_rows_2['Term 3'] == w3]
            if not trigram_row.empty:
                trigram_freq = trigram_row['Frequency of Trigram'].iloc[0]
                trigramProbability.append((trigram_freq + 1) / (context_freq_2 + 1))
            else:
                trigramProbability.append(1 / (context_freq_2 + 1))
        else:
            trigramProbability.append(1 / bigram_count)
    print(trigramProbability)
    return trigramProbability

def calculateProbabilities(model_filenames_str, test_sentence):
    nlp = spacy.load('es_core_news_sm')
    model_filenames = [name.strip() for name in model_filenames_str.split(',')]
    models = loadModels(model_filenames)


    test_sentence = "$ " + test_sentence + " #"
    words = [token.text for token in nlp(test_sentence)]
    print(words)

    sentenceProbability = [1] * len(models)

    for i, filename in enumerate(model_filenames):
        model_type = filename.split('_')[0].lower()
        print(model_type)

        if model_type == 'bigram':
            for j in range(len(words) - 1):
                print(f'{words[j]} \t {words[j+1]}\n')
                probabilities = getBigramProbabilities(words[j], words[j + 1], [models[i]])
                sentenceProbability[i] *= probabilities[0]
        elif model_type == 'trigram':
            for j in range(len(words) - 2):
                print(f'{words[j]}\t{words[j+1]}\t{words[j+2]}\n')
                probabilities = getTrigramProbabilities(words[j], words[j + 1], words[j + 2], [models[i]])
                sentenceProbability[i] *= probabilities[0]
        else:
            print(f"El archivo '{filename}' no es un modelo válido (bigram/trigram).")

    formattedProbabilities = [(model_filenames[i], prob) for i, prob in enumerate(sentenceProbability)]
    sortedProbabilities = sorted(formattedProbabilities, key=lambda x: x[1], reverse=True)

    return sortedProbabilities

