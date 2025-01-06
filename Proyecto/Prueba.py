from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments
from datasets import load_dataset
from rouge_score import rouge_scorer

modelo_base = "mrm8488/bert2bert_shared-spanish-finetuned-summarization"
tokenizer = AutoTokenizer.from_pretrained(modelo_base)
model = AutoModelForSeq2SeqLM.from_pretrained(modelo_base)
xl_sum_dataset = load_dataset("csebuetnlp/xlsum", "spanish")

# Aquí limité los articulos considerados para tomar en cuenta lo que dijo el profe de solo tomar unos cientos y para que no se tarde tanto este pedo
train_subset = xl_sum_dataset['train'].select(range(400))
test_subset = xl_sum_dataset['test'].select(range(100))

def preprocess_data(examples):
    inputs = examples['text']
    labels = examples['summary']

    model_inputs = tokenizer(inputs, max_length=512, truncation=True, padding=True)
    labels = tokenizer(labels, max_length=128, truncation=True, padding=True)

    model_inputs['labels'] = labels['input_ids']
    return model_inputs

data = train_subset.map(preprocess_data, batched=True)

# Parte del fine tunning
training_args = TrainingArguments(
    output_dir="./results",          
    evaluation_strategy="no",     
    learning_rate=2e-5,              
    per_device_train_batch_size=4,   
    num_train_epochs=3,              
    weight_decay=0.01,               
)
trainer = Trainer(
    model=model,                         
    args=training_args,                  
    train_dataset=data,         
)
trainer.train()

def generate_summary(article):
    inputs = tokenizer(article, return_tensors="pt", max_length=512, truncation=True)
    outputs = model.generate(inputs["input_ids"], max_length=128, min_length=40, length_penalty=2.0)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def evaluate_model():
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    rouge_scores = {"rouge1": [], "rouge2": [], "rougeL": []}

    for example in test_subset:
        article = example['text']
        reference_summary = example['summary']
        generated_summary = generate_summary(article)

        scores = scorer.score(reference_summary, generated_summary)

        rouge_scores["rouge1"].append(scores['rouge1'].fmeasure)
        rouge_scores["rouge2"].append(scores['rouge2'].fmeasure)
        rouge_scores["rougeL"].append(scores['rougeL'].fmeasure)

    avg_rouge1 = sum(rouge_scores['rouge1']) / len(rouge_scores['rouge1'])
    avg_rouge2 = sum(rouge_scores['rouge2']) / len(rouge_scores['rouge2'])
    avg_rougeL = sum(rouge_scores['rougeL']) / len(rouge_scores['rougeL'])

    print("\nResultados Promedio de ROUGE:")
    print(f"ROUGE-1: {avg_rouge1:.4f}")
    print(f"ROUGE-2: {avg_rouge2:.4f}")
    print(f"ROUGE-L: {avg_rougeL:.4f}")

evaluate_model()