from flask import Flask,render_template,redirect,url_for
import numpy as np
import cv2
import os
import gdown
from flask import request
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename


MODEL_PATH = "Blood_Cell_PRED.h5"

if not os.path.exists(MODEL_PATH):
    gdown.download("https://drive.google.com/file/d/12Bbh3kaEBFsE2WLr3ymufSdu4bfi1WUr/view?usp=drive_link", MODEL_PATH, quiet=False)


app=Flask(__name__)
upload_folder='static/uploads'
app.config['upload_folder']=upload_folder

model=load_model('Blood_Cell_PRED.h5')
class_names=['BASOPHIL', 'EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

def predict_image(img_path):
    img=cv2.imread(img_path)
    img=cv2.resize(img,(224,224))
    img=img/255.0
    img=np.expand_dims(img,axis=0)
    prediction=model.predict(img)[0]
    class_idx=np.argmax(prediction)
    label=class_names[class_idx]
    confidence=prediction[class_idx]
    return label,confidence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])

def predict():
    if 'file' not in request.files:
        return 'No File uploaded'
    file=request.files['file']
    if file.filename == '':
        return 'Empty Filename'
    filename=secure_filename(file.filename)
    file_path=os.path.join(app.config['upload_folder'],filename)
    file.save(file_path)

    label,confidence=predict_image(file_path)
    return render_template('result.html',label=label,confidence=confidence,filename=filename)

# if __name__=='__main__':
#     app.run(debug=True)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
