from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from rouge_score import rouge_scorer

#Cargar modelo y tokenizador
modelo = "mrm8488/bert2bert_shared-spanish-finetuned-summarization"
tokenizer = AutoTokenizer.from_pretrained(modelo)
model = AutoModelForSeq2SeqLM.from_pretrained(modelo)

#Ejemplos de textos con sus resumenes de muestra
ejemplos = [
    {
        "articulo": "El cambio climático se refiere a los cambios a largo plazo en las temperaturas y patrones climáticos. Estos cambios pueden ser naturales, pero desde el siglo XIX, las actividades humanas han sido el principal motor del cambio climático, principalmente debido a la quema de combustibles fósiles como el carbón, el petróleo y el gas. La quema de estos materiales produce gases de efecto invernadero que atrapan el calor en la atmósfera, lo que provoca el calentamiento global.",
        "resumen_referencia": "El cambio climático implica alteraciones en temperaturas y climas, mayormente por actividades humanas como la quema de combustibles fósiles que generan calentamiento global."
    },
    {
        "articulo": "El fútbol es uno de los deportes más populares del mundo. Se juega en más de 200 países y tiene millones de seguidores. Los partidos consisten en dos equipos de once jugadores que intentan marcar goles en la portería contraria. Las reglas del fútbol son establecidas por la FIFA, el organismo rector de este deporte. Cada año, torneos como la Copa del Mundo reúnen a los mejores equipos y jugadores, generando gran entusiasmo entre los fanáticos.",
        "resumen_referencia": "El fútbol, regulado por la FIFA, es un deporte popular en más de 200 países donde equipos de once jugadores buscan anotar goles."
    },
    {
        "articulo": "La inteligencia artificial (IA) es una rama de la informática que busca crear sistemas capaces de realizar tareas que normalmente requieren inteligencia humana, como el reconocimiento de voz, la toma de decisiones y la traducción de idiomas. La IA se ha convertido en una herramienta clave en diversas industrias, desde la salud hasta la automoción, y promete revolucionar aún más la forma en que vivimos y trabajamos.",
        "resumen_referencia": "La IA desarrolla sistemas para tareas que requieren inteligencia humana y transforma industrias como la salud y la automoción."
    }
]

ejemplo = ejemplos[0]
# ejemplo = ejemplos[1]
# ejemplo = ejemplos[2]

# Función para generar un resumen
def generate_summary(article):
    inputs = tokenizer(article, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs["input_ids"], max_length=128, min_length=40, length_penalty=2.0, num_beams=4)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

articulo = ejemplo["articulo"]
resumen_referencia = ejemplo["resumen_referencia"]

generated_summary = generate_summary(articulo)

print("Artículo original:")
print(articulo[:500]) 
print("\nResumen de referencia:")
print(resumen_referencia)
print("\nResumen generado:")
print(generated_summary)

#Evalua con ROUGE
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
scores = scorer.score(resumen_referencia, generated_summary)

print("\nPuntuaciones ROUGE:")
for key, value in scores.items():
    print(f"{key}: Recall: {value.recall:.2f}, Precision: {value.precision:.2f}, F1: {value.fmeasure:.2f}")
