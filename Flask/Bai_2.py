from flask import Flask, flash, request, redirect, url_for, render_template
import os
from werkzeug.utils import secure_filename
import cv2 as cv
import keras

model = keras.models.load_model('train_model.pb')

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'

app.secret_key = "secret key"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif', 'jfif'])

def prepare(filepath):
    img_size = 244
    img_array = cv.imread(filepath, cv.IMREAD_GRAYSCALE)
    new_array = cv.resize(img_array, (img_size, img_size))
    new_array = new_array / 255.0
    return new_array.reshape(-1, img_size, img_size, 1)

def predict(path_img):
    prediction = model.predict([prepare('{}'.format(path_img))])
    a = prediction[0]
    c = 0
    if a[0] >= 0.9:
        b = 'Mèo'
        c = a[0]
    elif a[1] >= 0.9:
        b = 'Chó'
        c = a[1]
    elif a[2] >= 0.9:
        b = 'Gấu trúc'
        c = a[2]
    else:
        b = 'Unknow'
        c = a[3]
    return b,c

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.')[1] in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/', methods=['GET','POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(path)
        flash('Image successfully uploaded and displayed below')
        b, c = predict(path)
        return render_template('index.html', filename=filename, b= b, c =c*100 )
    else:
        flash('Allowed image types are - png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == "__main__":
    app.run()
