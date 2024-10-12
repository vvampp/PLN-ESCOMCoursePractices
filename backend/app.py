from flask import Flask, request, render_template, redirect, url_for, flash
import os
import pandas as pd
from vectorization import vectorizeAll
from normalization import normalize

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

                flash('Vectorización completada exitosamente.')
                return redirect(url_for('home'))
            else:
                flash('El archivo debe ser un CSV.', 'error')
                return redirect(url_for('home'))

        except Exception as e:
            flash(f'Ocurrió un error durante la vectorización: {str(e)}', 'error')
            return redirect(url_for('home'))

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

            if file and file.filename.endswith('.csv'):
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
                file.save(filepath) 
                normalize(filepath)
                os.remove(filepath)

                flash('Normalización completada exitosamente.', 'message')
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