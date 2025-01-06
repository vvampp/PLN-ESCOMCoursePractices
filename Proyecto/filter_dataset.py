from transformers import pipeline
from datasets import load_dataset

# Configurar el pipeline para GPU
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli", device=0)

# Cargar el dataset
xl_sum_dataset = load_dataset("csebuetnlp/xlsum", "spanish")

# Definir etiquetas
labels = ["deportes", "finanzas", "tecnología", "salud", "cultura"]

# Comenzar con un batch_size de 8
batch_size = 64  # Ajusta según tu memoria

# Función para aplicar la clasificación en lotes
def filter_sports_batch(batch):
    texts = batch["summary"]
    results = classifier(texts, candidate_labels=labels, batch_size=batch_size)
    
    # Filtrar si la probabilidad de deportes es suficientemente alta
    sports_labels = []
    for result in results:
        if result["scores"][0] > 0.8:  # Ajusta el umbral según lo que necesites
            sports_labels.append(result["labels"][0])
        else:
            sports_labels.append("no_deportes")  # Si no tiene una probabilidad alta, asigna una etiqueta diferente
    
    return {
        "labels": sports_labels,
        "summary": batch["summary"]
    }


# Aplicar el filtro al dataset
sports_dataset = xl_sum_dataset["train"].map(filter_sports_batch, batched=True)

# Filtrar solo los artículos de deportes
sports_dataset = sports_dataset.filter(lambda example: example['labels'] == 'deportes')

sports_dataset.to_csv("sports_dataset.tsv",sep="\t", index=False)

print(f"Artículos relacionados con deportes: {len(sports_dataset)}")
