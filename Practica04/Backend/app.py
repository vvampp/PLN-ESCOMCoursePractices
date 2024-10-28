from flask import Flask, request, render_template, redirect, url_for, flash, jsonify, session
import os

# Importar funciones
from ModelosDeLenguage.generarModelos import generar_modelo
from TextoPredictivo.predecirTexto import get_palabras_probables
from GeneracionDeTexto.generarTexto import generar_texto
from ProbabilidadCondicional.calcularProbabilidad import calculateProbabilities
app = Flask(__name__, template_folder='../Frontend/templates', static_folder='../Frontend/static', static_url_path='/static')

app.secret_key = 'your_secret_key'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'ModelosDeLenguage', 'TokenizedCorporea')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/modelos_lenguaje', methods=['POST'])
def modelos_lenguaje():
    if request.method == 'POST':
        if 'file-input-corpus' not in request.files:
            flash("No se seleccionó ningún archivo", "error")
            return redirect(url_for('language_models'))

        file = request.files['file-input-corpus']
        ngram_type = request.form.get('n-gram-type')

        if file and file.filename.endswith('.tsv'):
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)
            tsv_file = file.filename
            
            if ngram_type == 'bigram':
                generar_modelo(tsv_file, 'bigram')
                flash('Modelado de bigramas completado!')
            elif ngram_type == 'trigram':
                generar_modelo(tsv_file, 'trigram')
                flash('Modelado de trigramas completado!')

            return redirect(url_for('home'))

    return redirect(url_for('home'))


@app.route('/prediccion_texto', methods=['POST'])
def prediccion_texto():
    if request.method == 'POST':
        if 'file-input-corpus' not in request.files:
            flash("No se seleccionó ningún archivo", "error")
            return redirect(url_for('predictiveText'))

        file = request.files['file-input-corpus']
        # Si envian archivo y palabras, pero no la palabra probable a  agregar, significa que es la primera iteracion
        # el proceso es de prediccion es
        # 1. Se recibe archivo tsv
        # 2. Se recibe palabras para iniciar prediccion
        # 3. Se determina si es bigrama o trigrama con base a la cantidad de palabras
        # 4. Se obtienen las palabras probables
        # 5. La pagina se refresca cuando se le da al boton de 'Next Word', esta pagina refrescada imprime
        # 5.1. En texto generado imprime las palabras que estaban ingresadas,
        # 5.2. En palabras probables se imprimen las palabras recuperadas
        # 6. El usuario escoge la nueva palabra y presiona el boton addWord y pasamos al segundo if
        print(f'filename: {file.filename}, words: {request.form.get("words")}, probable-words: {request.form.get("probable-words")}')
        if file and file.filename.endswith('.tsv') and request.form.get('words') and not request.form.get('probable-words'):
            print('entro a primera iteracion')
            # Parametros para la prediccion
            tsv_file = file.filename
            # Para el posterior renderizado
            session['tsv_file'] = tsv_file
            words = request.form.get('words')
            # Feature se determinar segun el numero de palabras en la cadena words
            feature = "bi" if len(words.split()) == 1 else "tri"

            print(f'Archivo: {tsv_file}, Palabras: {words}, Feature: {feature}')
            probable_words = get_palabras_probables(tsv_file, feature, words)
            print(probable_words)

            # probable_words es una lista que no es seguro que tenga las 4 palabras, por lo que se debe
            # hacer un chequeo para saber si se tienen las 4 palabras
            print(f'Cantidad de palabras: {len(probable_words)}')
            if len(probable_words) < 4:
                # Si no tiene las 4 palabras, se debe llenar con espacios
                for i in range(4 - len(probable_words)):
                    probable_words.append("")

            print(probable_words)
            print(f'Datos enviados a frontend: probable_word1: {probable_words[0]}, '
                  f'probable_word2: {probable_words[1]}, '
                  f'probable_word3: {probable_words[2]}, '
                  f'probable_word4: {probable_words[3]}, '
                  f'predicted_text: {words}, tsv_file: {tsv_file}, '
                  f'feature: {feature}')

            # Antes de enviar a frontend, se debe guardar en session la variable predicted_text
            session['predicted_text'] = words
            session['feature'] = feature

            # enviar a frontend
            # los datos enviados al front son:
            # 1. Las palabras probables
            # 2. El texto generado hasta el momento
            # 3. El archivo tsv (solo el nombre)
            return render_template('PredictiveText.html', probable_word1 = probable_words[0],
                                    probable_word2 = probable_words[1],
                                    probable_word3 = probable_words[2],
                                    probable_word4 = probable_words[3],
                                    predicted_text= words,
                                    tsv_file = session['tsv_file'],
                                    feature = feature
                                    )

        # Si se envia la palabra probable, significa que es la segunda o la n-esima iteracion
        # El proceso es el siguiente
        # 1. Se recibe la palabra probable
        # 2. Se recibe el texto generado hasta el momento (predicted_text)
        # 3. Se concatena la palabra probable al texto generado
        # 4. Se obtienen las nuevas palabras probables
        if request.form.get('probable-words'):
            print('entro a segunda o n iteracion')

            tsv_file = session['tsv_file']
            new_word = request.form.get('probable-words')
            # Primero se recupera el texto generado hasta el momento si es que hay
            # de la variable session
            predicted_text = session['predicted_text']
            feature = session['feature']
            predicted_text = predicted_text + " " + new_word

            print(f'Archivo: {tsv_file}, Palabras: {predicted_text}, Feature: {feature}')

            # Dependiendo si es bigrama o trigrama, se obtienen las palabras probables
            if feature == "bi":
                latest_word = predicted_text.split()[-1]
                probable_words = get_palabras_probables(tsv_file, feature, latest_word)
            else:
                latest_words = " ".join(predicted_text.split()[-2:])
                probable_words = get_palabras_probables(tsv_file, feature, latest_words)

            # probable_words es una lista que no es seguro que tenga las 4 palabras, por lo que se debe
            # hacer un chequeo para saber si se tienen las 4 palabras
            # primero revisemos que probable_words sea diferentes del tipo NoneType
            if not probable_words:
                return render_template('PredictiveText.html', probable_word1 = "",
                                    probable_word2 = "",
                                    probable_word3 = "",
                                    probable_word4 = "",
                                    predicted_text= predicted_text,
                                    tsv_file = session['tsv_file'],
                                    feature = feature
                                    )
            print(f'Cantidad de palabras: {len(probable_words)}')
            if len(probable_words) < 4:
                # Si no tiene las 4 palabras, se debe llenar con espacios
                for i in range(4 - len(probable_words)):
                    probable_words.append("")

            # antes de enviar a front, actualiza la variable de session
            session['predicted_text'] = predicted_text

            # enviar a frontend y el mismo proceso se repite
            return render_template('PredictiveText.html', probable_word1 = probable_words[0],
                                    probable_word2 = probable_words[1],
                                    probable_word3 = probable_words[2],
                                    probable_word4 = probable_words[3],
                                    predicted_text= predicted_text,
                                    tsv_file = session['tsv_file'],
                                    feature = feature
                                    )

    print('entro a nada xd')
    return redirect(url_for('predictiveText'))

@app.route('/generacion_texto', methods=['POST'])
def generacion_texto():
    if request.method == 'POST':
        if 'file-input-corpus' not in request.files:
            flash("No se seleccionó ningún archivo", "error")
            return redirect(url_for('home'))

        file = request.files['file-input-corpus']
        if file and file.filename.endswith('.tsv'):
            tsv_file = file.filename

            # Como el archivo ya esta guardado en el servidor, solo tenemos que recuperar el archivo
            # para el posterior renderizado
            session['tsv_file'] = tsv_file

            feature = "bi" if "bigram" in tsv_file.split('_')[0] else "tri"
            words = '$'

            generated_text = generar_texto(session['tsv_file'], feature, words)
            print(generated_text)
            generated_text = generated_text[1:-1] if len(generated_text) > 2 else ""

            return render_template('TextGeneration.html',
                                   generated_text=generated_text,
                                   tsv_file=session['tsv_file']
                                   )


        elif session['tsv_file']:
            # Obtener archivo tsv del value que hay en session
            tsv_file = session['tsv_file']
            feature = "bi" if "bigram" in tsv_file.split('_')[0] else "tri"
            words = '$'

            generated_text = generar_texto(session['tsv_file'], feature, words)
            print(generated_text)
            generated_text = generated_text[1:-1] if len(generated_text) > 2 else ""

            return render_template('TextGeneration.html',
                                   generated_text=generated_text,
                                   tsv_file=session['tsv_file']
                                   )

    return redirect(url_for('textGeneration'))

@app.route('/probabilidad', methods=['POST'])
def probabilidad():
    model_names = request.form.get('model_names', '') 
    test_sentence = request.form.get('test_sentence', '')
    
    print('Oracion: ' + test_sentence)

    probabilities = calculateProbabilities(model_names, test_sentence)

    print(f"Probabilidades calculadas: {probabilities}")

    probabilities_dict = [{"model": name, "probability": prob} for name, prob in probabilities]

    return jsonify(probabilities_dict)


@app.route('/predictive_text')
def predictive_text(
        probable_word1,
        probable_word2,
        probable_word3,
        probable_word4,
        words,
        tsv_file,
        feature
):
    return render_template('PredictiveText.html',
                           probable_word1=probable_word1,
                           probable_word2=probable_word2,
                           probable_word3=probable_word3,
                           probable_word4=probable_word4,
                           predicted_text=words,
                           tsv_file=tsv_file,
                           feature=feature
                           )


@app.route('/')
def home():
    return render_template('LanguageModels.html')

@app.route('/conditionalProb')
def conditionalProb():
    return render_template('ConditionalProbability.html')

@app.route('/predictiveText')
def predictiveText():
    return render_template('PredictiveText.html')

@app.route('/textGeneration')
def textGeneration():
    return render_template('TextGeneration.html')

if __name__ == '__main__':
    app.run(debug=True)
