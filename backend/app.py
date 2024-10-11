from flask import Flask, request, render_template, redirect, url_for, flash
import os
import pandas as pd
from vectorization import main

app = Flask(__name__, template_folder='../templates')

app.secret_key = 'your_secret_key'  
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET','POST'])
def vectorizeAPI ():
    if request.method == 'POST':
        if 'file-input-corpus' not in request.files:
            flash("No se seleccionó ningún archivo", "error")
            return redirect(url_for('vectorizeAPI'))
        
        file = request.files['file-input-corpus']

        if file.filename == '':
            flash("No se seleccionó ningún archivo", "error")
            return redirect(url_for('vectorizeAPI'))
        
        if file and file.filename.endswith('.csv'):
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            df = pd.read_csv(filepath, sep='\t')
            main(df)
            os.remove(filepath)
            flash('Vectorización completada exitosamente.')
            return redirect(url_for('vectorizeAPI'))
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
