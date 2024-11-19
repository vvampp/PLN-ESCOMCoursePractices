import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from nltk.corpus import wordnet
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

# Descargar recursos necesarios de nltk
import nltk
# nltk.download('wordnet')
# nltk.download('punkt')
# nltk.download('stopwords')

df = pd.read_excel("Rest_Mex_2022.xlsx")
df["feature"] = df["Title"].astype(str) + " " + df["Opinion"].astype(str)

# Valoers antes del balanceo
print("Distribución original de clases:")
print(df["Polarity"].value_counts())
print("Porcentaje antes del balanceo:")
print(df["Polarity"].value_counts(normalize=True) * 100)


# Separación en Training y Test (80%-20%)
train_data, test_data = train_test_split(df, test_size=0.2, stratify=df["Polarity"], random_state=0, shuffle=True)

# Aumento de datos
train_data_augmented = train_data.copy()

# Balanceo del conjunto de entrenamiento
class_counts = train_data_augmented["Polarity"].value_counts()
average_class_size = class_counts.mean()

#Margenes de balanceo
balance_factors = {
    5: (0.35, 0.40), 
    4: (0.25, 0.30), 
    3: (0.25, 0.30), 
    2: (0.20, 0.25), 
    1: (0.10, 0.15)  
}

balanced_train_data = []

for polarity, count in class_counts.items():
    class_data = train_data_augmented[train_data_augmented["Polarity"] == polarity]
    
    min_factor, max_factor = balance_factors[polarity]
    factor = min_factor + (max_factor - min_factor) * (count / class_counts.max()) 
    
    target_size = int(average_class_size * factor)
    
    if count > target_size:
        # Submuestreo para clases mayoritarias
        class_data = resample(class_data, replace=False, n_samples=target_size, random_state=42)
    elif count < target_size:
        # Sobremuestreo para clases minoritarias
        class_data = resample(class_data, replace=True, n_samples=target_size, random_state=42)
    
    balanced_train_data.append(class_data)

balanced_train_data = pd.concat(balanced_train_data)

#Generación de representaciones
vectorizer_binary = CountVectorizer(binary=True)
vectorizer_frequency = CountVectorizer()
vectorizer_tfidf = TfidfVectorizer()

# Ajuste y transformación de datos para cada representación
X_train_binary = vectorizer_binary.fit_transform(balanced_train_data["feature"])
X_train_frequency = vectorizer_frequency.fit_transform(balanced_train_data["feature"])
X_train_tfidf = vectorizer_tfidf.fit_transform(balanced_train_data["feature"])

X_test_binary = vectorizer_binary.transform(test_data["feature"])
X_test_frequency = vectorizer_frequency.transform(test_data["feature"])
X_test_tfidf = vectorizer_tfidf.transform(test_data["feature"])

balanced_train_data.to_csv("balanced_train_data.csv", index=False)
test_data.to_csv("test_data.csv", index=False)

# Valores después del balanceo
print("Distribución después del balanceo:")
print(balanced_train_data["Polarity"].value_counts())
print("Porcentaje después del balanceo:")
print(balanced_train_data["Polarity"].value_counts(normalize=True) * 100)

print("Representaciones del corpus creadas:")
print(f"Representación binarizada: {X_train_binary.shape}")
print(f"Representación frecuencia: {X_train_frequency.shape}")
print(f"Representación TF-IDF: {X_train_tfidf.shape}")

print("Creación de dataset completada.")