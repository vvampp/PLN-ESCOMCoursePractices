import pandas as pd
import re
import spacy
import nltk
from collections import Counter
import os.path

nlp = spacy.load('es_core_news_sm')

def readTokenizedCorpus(tsv_file):
    tsv_folder = os.path.join(os.getcwd(), 'ModelosDeLenguage', 'TokenizedCorporea')
    tsv_filepath = os.path.join(tsv_folder, tsv_file)

    try:
        df = pd.read_csv(tsv_filepath, sep='\t', encoding='utf-8')
        return df['Documents'].tolist() 
    except Exception as e:
        print(f"An exception occurred: {e}")
        return

def calculateFrequency(sentencized_corpus, bigram_destiny_tsv_file=None, trigram_destiny_tsv_file=None, ngram_type=None):

    language_model_folder = os.path.join(os.getcwd(), 'ModelosDeLenguage', 'LanguageModels')

    unigram_freq = Counter()
    bigram_freq = Counter()
    trigram_freq = Counter()

    docs = list(nlp.pipe(sentencized_corpus, batch_size=50))

    for doc in docs:
        tokens = [token.text for token in doc]
        bigrams = list(nltk.bigrams(tokens))
        trigrams = list(nltk.trigrams(tokens))

        unigram_freq.update(tokens)
        bigram_freq.update(bigrams)
        trigram_freq.update(trigrams)

    if ngram_type == 'bigram':
        calculateBigram(bigram_freq, unigram_freq, language_model_folder, bigram_destiny_tsv_file)
    elif ngram_type == 'trigram':
        calculateTrigram(trigram_freq, bigram_freq, language_model_folder, trigram_destiny_tsv_file)

def calculateBigram(bigram_freq,unigram_freq,language_model_folder, destiny_tsv_file):
    # destiny_tsv_file = 'bigram_language_model_adair.tsv'
    rows = []
    bigram_probabilities = {}
    for bigram in bigram_freq:
        w1, w2 = bigram
        bigram_probabilities[bigram] = bigram_freq[bigram]/unigram_freq[w1]
        rows.append( {
            "Term 1": w1,
            "Term 2": w2,
            "Frequency of Bigram": bigram_freq[bigram],
            "Frequency of context": unigram_freq[w1],
            "Conditional Probability of Bigram":bigram_probabilities[bigram]
        })
    sorted_rows = sorted(rows, key = lambda x: x["Frequency of Bigram"], reverse=True)
    bigram_tsv = pd.DataFrame(sorted_rows)
    language_model_filepath = os.path.join(language_model_folder,destiny_tsv_file)
    bigram_tsv.to_csv(language_model_filepath,sep='\t',index=False)

def calculateTrigram(trigram_freq,bigram_freq,language_model_folder, destiny_tsv_file):
    # destiny_tsv_file = 'trigram_language_model_adair.tsv'
    rows = []
    trigram_probabilities = {}
    for trigram in trigram_freq:
        w1, w2, w3 = trigram
        trigram_probabilities[trigram] = trigram_freq[trigram]/bigram_freq[(w1,w2)]
        rows.append({
            "Term 1": w1,
            "Term 2": w2,
            "Term 3": w3,
            "Frequency of Trigram": trigram_freq[trigram],
            "Frequency of Context": bigram_freq[(w1,w2)],
            "Conditional Probability of Trigram":  trigram_probabilities[trigram]                                            
        })
    sorted_rows = sorted(rows, key = lambda x: x["Frequency of Trigram"], reverse=True)
    trigram_tsv = pd.DataFrame(sorted_rows)
    language_model_filepath = os.path.join(language_model_folder,destiny_tsv_file)
    trigram_tsv.to_csv(language_model_filepath,sep='\t',index=False)

def generar_modelo(tsv_file, ngram_type):
    sentencized_corpus = readTokenizedCorpus(tsv_file)

    if ngram_type == 'bigram':
        bigram_destiny_tsv_file = f"bigram_language_model_{tsv_file.split('.')[0]}.tsv"
        calculateFrequency(sentencized_corpus, bigram_destiny_tsv_file=bigram_destiny_tsv_file, ngram_type='bigram')
    elif ngram_type == 'trigram':
        trigram_destiny_tsv_file = f"trigram_language_model_{tsv_file.split('.')[0]}.tsv"
        calculateFrequency(sentencized_corpus, trigram_destiny_tsv_file=trigram_destiny_tsv_file, ngram_type='trigram')
