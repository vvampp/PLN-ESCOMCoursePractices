from flask import Flask, request, render_template, redirect, url_for, flash
import os
import pandas as pd

# from backend.analyzer import vectorizeTest, cosine_similarity
from corpusVectorization import vectorizeAll
from testAnalyzer import testAnalyzer      # testAnalyzer(test.txt,compare_element,vector_type,feature_type) -> NORMALIZACIÓN -> VECTORIZACIÓN DINÁMICA

app = Flask(__name__, template_folder='../templates')

app.secret_key = 'your_secret_key'  
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/vectorizar', methods=['GET', 'POST'])
def vectorizeAPI():
    if request.method == 'POST':
        try:
            if 'file-input-corpus' not in request.files:
                flash("No se seleccionó ningún archivo", "error")
                return redirect(url_for('home'))

            file = request.files['file-input-corpus']

            if file.filename == '':
                flash("No se seleccionó ningún archivo", "error")
                return redirect(url_for('home'))

            if file and file.filename.endswith('.csv'):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath)

                df = pd.read_csv(filepath, sep='\t')

                vectorizeAll(df)

                os.remove(filepath)

                flash('Vectorización del corpus completada exitosamente.')
                return redirect(url_for('home'))
            else:
                flash('El archivo debe ser un CSV.', 'error')
                return redirect(url_for('home'))

        except Exception as e:
            flash(f'Ocurrió un error durante la vectorización: {str(e)}', 'error')
            return redirect(url_for('home'))

    return render_template('index.html')

@app.route('/analizar', methods=['POST'])
def analizar_documento():
    if request.method == 'POST':
        try:
            if 'file-input-test' not in request.files:
                flash("No se seleccionó ningún archivo", "error")
                return redirect(url_for('home'))

            file = request.files['file-input-test']

            if file.filename == '':
                flash("No se seleccionó ningún archivo", "error")
                return redirect(url_for('home'))

            if file and file.filename.endswith('.txt'):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath) 

                # obtener parámetors
                vector_type = request.form.get('vector-type')
                feature_type = request.form.get('feature-type')
                compare_element = request.form.get('compare-element')

                if not all ([vector_type, feature_type, compare_element]):
                    flash("Campos en el formulario faltantes", "error")
                    return redirect(url_for('home'))
                
                testAnalyzer ( test_txt_file = filepath, vector_type=vector_type, feature_type=feature_type, compare_element=compare_element)

                flash('Proceso completado existosamente, revisar consola', 'message')
                return redirect(url_for('home')) 
            else:
                flash('El archivo debe ser un CSV.', 'error')
                return redirect(url_for('home')) 

        except Exception as e:
            flash(f'Ocurrió un error durante la normalización: {str(e)}', 'error')
            return redirect(url_for('home'))  

    return redirect(url_for('home')) 

@app.route('/')
def home():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)