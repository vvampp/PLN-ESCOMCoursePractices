import spacy 
import pandas as pd
import os


def normalizeTest(test_file_content):

    destiny_txt_file = 'normalized_test.txt'
    nlp = spacy.load('es_core_news_sm')

    content = nlp(test_file_content)
    nContent = ""

    for token in content:
        if not token.pos_ in ["DET","ADP", "CCONJ", "SCONJ", "PRON"]:
            nContent = nContent + token.lemma_ + " "

    output_folder = os.path.join(os.getcwd(), 'normalized_test')
    os.makedirs(output_folder, exist_ok=True)
    filepath = os.path.join(output_folder, destiny_txt_file)

    with open(filepath, "w") as file:
        print(nContent, file=file)



def normalize (test_txt_file):

    try:
        with open(test_txt_file, 'r') as file:
            test_file_content = file.read().rstrip() 
    except Exception as e:
        print(f"An exception ocurred: {e}")

    normalizeTest(test_file_content) 

# if __name__ == "__main__":
#     main()