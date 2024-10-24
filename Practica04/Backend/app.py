from flask import Flask, request, render_template, redirect, url_for, flash

# Importar funciones
from Practica04.Backend.ModelosDeLenguage.generarModelos import generar_modelo
from Practica04.Backend.TextoPredictivo.predecirTexto import get_palabras_probables
from Practica04.Backend.GeneracionDeTexto.generarTexto import generar_texto

app = Flask(__name__, template_folder='../Frontend/templates')

app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/modelos_lenguaje', methods=['POST'])
def modelos_lenguaje():
    if request.method == 'POST':
        if 'file-input-corpus' not in request.files:
            flash("No se seleccionó ningún archivo", "error")
            return redirect(url_for('language_models'))

        file = request.files['file-input-corpus']
        if file and file.filename.endswith('.txt'):
            txt_file = file.filename
            generar_modelo(txt_file)
            flash('Modelado de lenguaje completado!.')
            return redirect(url_for('language_models'))

    return redirect(url_for('language_models'))


@app.route('/prediccion_texto', methods=['POST'])
def prediccion_texto():
    if request.method == 'POST':
        if 'file-input-corpus' not in request.files:
            flash("No se seleccionó ningún archivo", "error")
            return redirect(url_for('predictive_text'))

        file = request.files['file-input-corpus']
        # Si envian archivo y palabras, pero no la palabra probable a  agregar, significa que es la primera iteracion
        # el proceso es de prediccion es
        # 1 . Se recibe archivo tsv
        # 2. Se recibe palabras para iniciar prediccion
        # 3. Se determina si es bigrama o trigrama con base a la cantidad de palabras
        # 4. Se obtienen las palabras probables
        # 5. La pagina se refresca cuando se le da al boton de 'Next Word', esta pagina refrescada imprime
        # 5.1. En texto generado imprime las palabras que estaban ingresadas,
        # 5.2. En palabras probables se imprimen las palabras recuperadas
        # 6. El usuario escoge la nueva palabra y presiona el boton addWord y pasamos al segundo if
        if file and file.filename.endswith('.tsv') and request.form.get('id-words') and not request.form.get('form-probable-words'):
            # Parametros para la prediccion
            tsv_file = file.filename
            words = request.form.get('id-words')
            # Feature se determinar segun el numero de palabras en la cadena words
            feature = "bi" if len(words.split()) == 1 else "tri"

            probable_words = get_palabras_probables(tsv_file, feature, words)

            # enviar a frontend
            return redirect(url_for('predictive_text', probable_word1 = probable_words[0],
                                    probable_word2 = probable_words[1],
                                    probable_word3 = probable_words[2],
                                    probable_word4 = probable_words[3],
                                    predicted_text= words,
                                    tsv_file = tsv_file
                                    ))


        if request.form.get('form-probable-words'):
            new_word = request.form.get('form-probable-words')
            predicted_text = request.form.get('predicted-text')

            predicted_text = predicted_text + " " + new_word



    return redirect(url_for('predictive_text'))

@app.route('/generacion_texto', methods=['POST'])
def generacion_texto():
    if request.method == 'POST':
        if 'file-input-corpus' not in request.files:
            flash("No se seleccionó ningún archivo", "error")
            return redirect(url_for('generate_text'))

        file = request.files['file-input-corpus']
        if file and file.filename.endswith('.tsv'):
            # Parametros para la generacion
            tsv_file = file.filename
            words = request.form.get('id-words')
            # Feature se determinar segun el numero de palabras en la cadena words
            feature = "bi" if len(words.split()) == 1 else "tri"

            generated_text = generar_texto(tsv_file, feature, words)

            return redirect(url_for('generate_text', generated_text = generated_text))

    return redirect(url_for('generate_text'))

@app.route('/probabilidad_condicional', methods=['POST'])
def probabilidad_condicional():

    return redirect(url_for('home'))

@app.route('/language_models')
def language_models():
    return render_template('LanguageModels.html')

@app.route('/predictive_text')
def predictive_text(
        probable_words,
        words,
        tsv_file
):
    return render_template('PredictiveText.html',
                           probable_word1=probable_words[0],
                           probable_word2=probable_words[1],
                           probable_word3=probable_words[2],
                           probable_word4=probable_words[3],
                           predicted_text=words,
                           tsv_file=tsv_file
                           )

@app.route('/generate_text')
def generate_text(generated_text):
    return render_template('TextGeneration.html',
                           generated_text = generated_text
                           )

if __name__ == '__main__':
    app.run(debug=True)
