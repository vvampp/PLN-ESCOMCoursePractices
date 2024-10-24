import pandas as pd
import re
import spacy
import nltk
from collections import Counter
import os.path

nlp = spacy.load('es_core_news_sm')

def readChat(txt_file):
    txt_folder = os.path.join(os.getcwd(), 'RawCorporea')
    txt_filepath = os.path.join(txt_folder,txt_file) 

    try:
        with open (txt_filepath, 'r', encoding='utf-8') as file:
            raw_corpus = '\n'.join([line.rstrip() for line in file])
            return raw_corpus

    except Exception as e:
        print(f"An exception occurred: {e}")
        return
    

def extractMessages(raw_corpus, pattern):
    messages = re.findall(pattern, raw_corpus)
    filtered_messages = [mesage for mesage in messages
                         if not re.search(r'\u200E',mesage)
                         and not re.search( r'https?://[A-Za-z0-9-]+\.+[A-Za-z]{2,}(?::\d{1,5})?(?:(?:/[\w\-\.~%]+)*)?(?:\?(?:[\w\-\.~%]+=[\w\-\.~%]*(?:&[\w\-\.~%]+=[\w\-.~%]*)*)?)?(?:#[\w\-.~%\/\+!\@\(\)\[\]\{\}]+)??', mesage)
                        and not re.search(r'<Media omitted>', mesage)
                         and not re.search(r'You deleted this message', mesage)
                         and not re.search(r'null', mesage)
                         and not re.search(r'file://(.+)', mesage)
                         ]
    return filtered_messages

def sentencize(messages, destiny_tsv_file):
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

def calculateFrequency(sentencized_corpus, bigram_destiny_tsv_file, trigram_destiny_tsv_file):
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

    calculateBigram(bigram_freq,unigram_freq,language_model_folder, bigram_destiny_tsv_file)
    calculateTrigram(trigram_freq,bigram_freq,language_model_folder, trigram_destiny_tsv_file)
    

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

def generar_modelo(file_txt):
    raw_corpus = readChat(file_txt)

    # El patron dependera del formato
    pattern = ""
    messages = extractMessages(raw_corpus, pattern)

    #El nombre del archivo se genera a partir del nombre del txt
    actor = obtener_nombre_actor(file_txt)
    destiny_tsv_file = f"tokenized_corpus_{actor}.tsv"
    sentencized_corpus = sentencize(messages, destiny_tsv_file)

    bigram_destiny_tsv_file = f"bigram_language_model_{actor}.tsv"
    trigram_destiny_tsv_file = f"trigram_language_model_{actor}.tsv"
    calculateFrequency(sentencized_corpus, bigram_destiny_tsv_file, trigram_destiny_tsv_file)

def obtener_nombre_actor(file_txt):
    nombre = file_txt.removeprefix("rawCorpus")
    nombre = nombre.removesuffix(".txt")
    return nombre

def main():
    raw_corpus = readChat("rawCorpusAdair.txt")

    # EL pattern depende del formato del archivo
    # pattern = r'\[\d{2}/\d{2}/\d{2}, \d{2}:\d{2}:\d{2}\] bambino: (.+)'
    pattern = r'[\d\d?/\d\d?/\d\d, \d\d?:\d\d( PM| AM)] - AdairG: (.+)'

    messages = extractMessages(raw_corpus, pattern)

    destiny_tsv_file = 'tokenized_corpus_Adair.tsv'
    sentencized_corpus = sentencize(messages, destiny_tsv_file)
    calculateFrequency(sentencized_corpus, 'bigram_language_model_adair.tsv', 'trigram_language_model_adair.tsv')


if __name__ == "__main__":
    main()
