import pandas as pd
import re
import spacy
import nltk
from collections import Counter
import os.path

nlp = spacy.load('es_core_news_sm')
#nltk.download('punkt')

def readChat():
    txt_file = 'rawCorpusBambino.txt'
    txt_folder = os.path.join(os.getcwd(), 'RawCorporea')
    txt_filepath = os.path.join(txt_folder,txt_file) 

    try:
        with open (txt_filepath, 'r', encoding='utf-8') as file:
            raw_corpus = '\n'.join([line.rstrip() for line in file])
            return raw_corpus

    except Exception as e:
        print(f"An exception occurred: {e}")
        return
    

def extractMessages(raw_corpus):
    pattern = r'\[\d{2}/\d{2}/\d{2}, \d{2}:\d{2}:\d{2}\] bambino: (.+)'
    messages = re.findall(pattern, raw_corpus)
    filtered_messages = [mesage for mesage in messages if not re.search(r'\u200E',mesage) and not re.search( r'https?:\/\/(?:(?:(?:[A-Za-z0-9-])+\.))+(?:[A-Za-z]){2,}(?::\d{1,5})?(?:(?:\/(?:[\w\-\.~%])+)*)?(?:\?(?:(?:[\w\-\.~%])+=(?:[\w\-\.~%])*(?:&(?:[\w\-\.~%])+=(?:[\w\-\.~%])*)*)?)?(?:#(?:[\w\-\.~%\/\+\!\@\(\)\[\]\{\}]+))??', mesage)]
    return filtered_messages


def sentencize(messages):
    destiny_tsv_file = 'tokenized_corpus_bambino.tsv'
    tsv_folder = os.path.join(os.getcwd(),'TokenizedCorporea')
    tsv_filepah = os.path.join(tsv_folder,destiny_tsv_file)

    df = pd.DataFrame()
    sentences = []

    if "sentencizer" not in nlp.pipe_names:
        nlp.add_pipe("sentencizer")

    docs = list(nlp.pipe(messages,batch_size=50))

    for doc in docs:
        for sent in doc.sents:
            sentence = sent.text
            sentences.append("$ "+sentence+" #")

    df = pd.DataFrame(sentences, columns=["Documents"])
    df.to_csv(tsv_filepah, sep='\t', index=False)
    return df

def calculateFrequency(sentencized_corpus):
    language_model_folder = os.path.join(os.getcwd(),'LanguageModels')

    unigram_freq = Counter()
    bigram_freq = Counter()
    trigram_freq = Counter()

    docs = list(nlp.pipe(sentencized_corpus['Documents'],batch_size=50))

    for doc in docs:
        tokens = [token.text for token in doc]
        bigrams = list(nltk.bigrams(tokens))
        trigrams = list(nltk.trigrams(tokens))

        unigram_freq.update(tokens)
        bigram_freq.update(bigrams)
        trigram_freq.update(trigrams)

    calculateBigram(bigram_freq,unigram_freq,language_model_folder)
    calculateTrigram(trigram_freq,bigram_freq,language_model_folder)
    

def calculateBigram(bigram_freq,unigram_freq,language_model_folder):
    destiny_tsv_file = 'bigram_language_model_bambino.tsv'
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
        


def calculateTrigram(trigram_freq,bigram_freq,language_model_folder):
    destiny_tsv_file = 'trigram_language_model_bambino.tsv'
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
  

def main():
    raw_corpus = readChat()
    messages = extractMessages(raw_corpus)
    sentencized_corpus = sentencize(messages)
    calculateFrequency(sentencized_corpus)
    

if __name__ == "__main__":
    main()