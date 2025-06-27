from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import cv2
import os
import gdown
import sqlite3
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

# Download model if not exists
MODEL_PATH = "Blood_Cell_PRED.h5"
if not os.path.exists(MODEL_PATH):
    gdown.download("https://drive.google.com/file/d/12Bbh3kaEBFsE2WLr3ymufSdu4bfi1WUr/view?usp=drive_link", MODEL_PATH, quiet=False)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'secret123'
app.config['upload_folder'] = 'static/uploads'

# Load the model
model = load_model(MODEL_PATH)
class_names = ['BASOPHIL', 'EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

# Initialize the database
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password TEXT NOT NULL)''')
    conn.commit()
    conn.close()

init_db()

# Prediction function
def predict_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    prediction = model.predict(img)[0]
    class_idx = np.argmax(prediction)
    label = class_names[class_idx]
    confidence = prediction[class_idx]
    return label, confidence

# Routes
@app.route('/')
def index():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']

    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
    user = c.fetchone()
    conn.close()

    if user:
        session['user'] = username
        return redirect(url_for('home'))
    else:
        return "Invalid credentials"

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        try:
            c.execute("INSERT INTO users (username, password) VALUES (?, ?)", (username, password))
            conn.commit()
            return redirect(url_for('index'))
        except sqlite3.IntegrityError:
            return "Username already exists"
        finally:
            conn.close()

    return render_template('register.html')

@app.route('/home')
def home():
    if 'user' not in session:
        return redirect(url_for('index'))
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No File uploaded'
    file = request.files['file']
    if file.filename == '':
        return 'Empty Filename'
    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['upload_folder'], filename)
    file.save(file_path)

    label, confidence = predict_image(file_path)
    return render_template('result.html', label=label, confidence=confidence, filename=filename)

@app.route('/logout')
def logout():
    session.pop('user', None)
    return redirect(url_for('index'))

# Run the app
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
