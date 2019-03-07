from flask import *
from app import app
import os
from flask import Flask, render_template, request



UPLOAD_FOLDER = os.path.basename('images')
app.config['images'] = UPLOAD_FOLDER



@app.route('/')
@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    file = request.files['image']
    f = os.path.join(app.config['images'], file.filename)

    # add your custom code to check that the uploaded file is a valid image and not a malicious file (out-of-scope for this post)
    file.save(f)

    return render_template('index.html')



