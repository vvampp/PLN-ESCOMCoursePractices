import csv
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os

# nltk.download('punkt')
# nltk.download('stopwords')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# Lematizador y las stop words específicas
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('spanish'))

stop_words.update([
    "el", "la", "los", "las", "un", "una", "unos", "unas", "al", "del",  # Artículos
    "a", "ante", "bajo", "cabe", "con", "contra", "de", "desde", "durante", "en", "entre", "hacia", "hasta", "mediante", "para", "por", "según", "sin", "so", "sobre", "tras",  # Preposiciones
    "y", "o", "u", "ni", "pero", "que", "si", "e",
    "aunque", "porque", "pues", "mientras", "sino", "cuando", "como", "donde", "tan", "tal", "más", "menos", "ya", "bien", "sea", "antes", "siempre", "entonces", "luego", "además", "incluso", "sin embargo",  # Conjunciones
    "me", "te", "se", "nos", "lo", "la", "le", "les", "lo", "él", "ella", "ellos", "ellas", "tú", "usted", "ustedes", "mío", "mía", "míos", "mías", "suyo", "suya", "suyos", "suyas"  # Pronombres
])

# Función para tokenización, eliminación de stop words y lematización
def procesar_texto(texto):
    # Añadir manejo de errores para caracteres extraños
    try:
        tokens = word_tokenize(texto.lower())
        tokens_procesados = []
        
        for token in tokens:
            if token not in stop_words:
                # Lematizar usando 'v' para verbos
                tokens_procesados.append(lemmatizer.lemmatize(token, pos='v'))
        
        print("Texto procesado:", tokens_procesados)  # Verificar tokens procesados
        return ' '.join(tokens_procesados)
    except Exception as e:
        print(f"Error procesando el texto: {e}")
        return texto  # Retornar el texto sin procesar en caso de error

def procesar_csv(filepath, output_filepath=None):
    noticias_procesadas = []
    
    try:
        with open(filepath, mode='r', newline='', encoding='utf-8') as archivo_csv:
            reader = csv.DictReader(archivo_csv)
            print("Cabeceras del CSV:", reader.fieldnames)

            for fila in reader:
                print("Fila original:", fila)  # Imprimir cada fila
                if 'Title' in fila and 'Content' in fila:
                    fila_procesada = fila.copy()
                    fila_procesada['Title'] = procesar_texto(fila['Title'])
                    fila_procesada['Content'] = procesar_texto(fila['Content'])
                    noticias_procesadas.append(fila_procesada)
                else:
                    print("Faltan las columnas necesarias en el CSV.")
    
    except FileNotFoundError:
        print(f"Archivo {filepath} no encontrado.")
        return

    # Define el archivo de salida
    if not output_filepath:
        output_filepath = os.path.splitext(filepath)[0] + '_normalized.csv'

    with open(output_filepath, mode='w', newline='', encoding='utf-8') as archivo_csv_salida:
        campos = ['Source', 'Title', 'Content', 'Section', 'URL', 'Date']
        writer = csv.DictWriter(archivo_csv_salida, fieldnames=campos)

        writer.writeheader()
        
        for noticia in noticias_procesadas:
            print("Escribiendo noticia procesada:", noticia)  # Verificar el contenido a escribir
            writer.writerow(noticia)

    print(f"Noticias procesadas guardadas en {output_filepath}")
