from flask import Flask, request, render_template, redirect, url_for, flash

# Importar funciones
from Practica04.ModelosDeLenguage.generarModelos import generar_modelo

app = Flask(__name__, template_folder='../templates')

app.secret_key = 'your_secret_key'
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def verificar_archivo(peticion):
    if 'file-input-corpus' not in peticion.files:
        flash("No se seleccionó ningún archivo", "error")
        return redirect(url_for('home'))

@app.route('/modelos_lenguaje', methods=['GET', 'POST'])
def modelos_lenguaje():
    # Recuperar todos los modelos en 'LanguageModels'
    if request.method == 'GET':
        print(f'GET: {request.args}')

    elif request.method == 'POST':
        verificar_archivo(request)

        file = request.files['file-input-corpus']
        if file and file.filename.endswith('.txt'):
            txt_file = file.filename
            generar_modelo(txt_file)
            flash('Modelado de lenguaje completado!.')
            return redirect(url_for('home'))

    print(f'POST: {request.files}')
    return redirect(url_for('home'))


@app.route('/prediccion_texto', methods=['GET', 'POST'])
def prediccion_texto():
    if request.method == 'POST':
        verificar_archivo(request)

        file = request.files['file-input-corpus']
        if file and file.filename.endswith('.txt'):
            txt_file = file.filename



            flash('Modelado de lenguaje completado!.')
            return redirect(url_for('home'))

    print(f'POST: {request.files}')
    return redirect(url_for('home'))



@app.route('/generacion_texto', methods=['POST'])
def generacion_texto():

    return redirect(url_for('home'))

@app.route('/probabilidad_condicional', methods=['POST'])
def probabilidad_condicional():

    return redirect(url_for('home'))

@app.route('/')
def home():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
