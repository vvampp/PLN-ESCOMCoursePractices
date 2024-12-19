from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from rouge_score import rouge_scorer

# 1. Cargar modelo y tokenizador
modelo = "mrm8488/bert2bert_shared-spanish-finetuned-summarization"
tokenizer = AutoTokenizer.from_pretrained(modelo)
model = AutoModelForSeq2SeqLM.from_pretrained(modelo)

# 2. Cargar el dataset XLSum para textos en español
dataset = load_dataset("csebuetnlp/xlsum", "spanish")

# Seleccionar un artículo para la prueba
ejemplo = dataset["train"][0]
articulo = ejemplo["text"]
resumen_referencia = ejemplo["summary"]

# 3. Función para generar un resumen
def generate_summary(article):
    inputs = tokenizer(article, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs["input_ids"], max_length=128, min_length=40, length_penalty=2.0, num_beams=4)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Generar el resumen
generated_summary = generate_summary(articulo)

# Mostrar resultados
print("Artículo original:")
print(articulo[:500]) 
print("\nResumen de referencia:")
print(resumen_referencia)
print("\nResumen generado:")
print(generated_summary)

# 4. Evaluar con ROUGE
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
scores = scorer.score(resumen_referencia, generated_summary)

print("\nPuntuaciones ROUGE:")
for key, value in scores.items():
    print(f"{key}: Recall: {value.recall:.2f}, Precision: {value.precision:.2f}, F1: {value.fmeasure:.2f}")
