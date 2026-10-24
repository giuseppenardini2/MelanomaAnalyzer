import numpy
import os
from flask import Flask, flash, request, redirect, url_for, send_from_directory, render_template
from werkzeug.utils import secure_filename
from backend import is_melanoma

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = 'abcdefg'

'''
Check if a file is allowed to be uploaded.
'''
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

'''
Manage the file uploading.
'''
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST': # Check if the request is post
        if 'file' not in request.files: # Check if the post request has the file part
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '': # If the user does not select a file, the browser submits an empty file without a filename
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename): # If the file has been uploaded and it is a valid file, calculate if the picture is a melanoma or not
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            image_url=url_for('get_file', name=filename)
            is_mel=is_melanoma(UPLOAD_FOLDER+'/'+filename) 
            return render_template('index.html', image_url=image_url, is_mel=is_mel)
    return render_template('index.html', image_url=None)

'''
Get the a file based on its filename.
'''
@app.route('/uploads/<name>')
def get_file(name):
    return send_from_directory(app.config["UPLOAD_FOLDER"], name)

if __name__ == '__main__':
    app.run()