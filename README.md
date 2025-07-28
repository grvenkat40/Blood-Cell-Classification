# 🧠 HematoVision: Advanced Blood Cell Classification Using Transfer Learning

HematoVision is a web-based deep learning project that uses **transfer learning (VGG16)** to classify human blood cells into five types: **Neutrophil**, **Lymphocyte**, **Monocyte**, **Eosinophil**, and **Basophil**. It is designed to assist medical professionals and labs by automating the process of blood cell classification from microscopic images.

---

## 📌 Features

- 🔍 Predicts blood cell type from an uploaded image
- 🧠 Utilizes VGG16 pre-trained model (Transfer Learning)
- 🌐 Web interface built with Flask + HTML/CSS
- ☁️ Dockerized and ready for deployment
- 🔐 User authentication (login/register)
- 📥 Model auto-downloaded via gdown from Google Drive

---
## 🔐 Login Page

![Login Screenshot]("D:\OneDrive\Pictures\Screenshots\Screenshot 2025-06-27 202515.png")


<img width="676" height="475" alt="Screenshot 2025-06-27 202515" src="https://github.com/user-attachments/assets/81deddd0-7f4c-47ff-ab98-c0d330bd715f" />


## 📂 Project Structure

```
HematoVision/
├── app.py                  # Flask application
├── Dockerfile              # Docker setup
├── requirements.txt        # Required Python packages
├── static/
│   ├── css/
│   │   └── style.css
│   └── uploads/
├── templates/
│   ├── index.html
│   ├── result.html
│   ├── login.html
│   └── register.html
├── hemato_vision_vgg16.h5  # Trained model (downloaded via gdown)
```

---

## ⚙️ Technologies Used

- **Frontend**: HTML, CSS
- **Backend**: Flask
- **Deep Learning**: TensorFlow, Keras, VGG16
- **Image Processing**: OpenCV
- **Deployment**: Docker + Render

---

## 🧪 Blood Cell Types Detected

- **Neutrophil**
- **Lymphocyte**
- **Monocyte**
- **Eosinophil**
- **Basophil**

---
## 🔗 Download Pretrained Model

This project requires a trained CNN model file to make predictions.

👉 [Download Blood_Cell_PRED.h5]("https://drive.google.com/file/d/12Bbh3kaEBFsE2WLr3ymufSdu4bfi1WUr/view?usp=drive_link")

Place the file in the root directory of the project (same location as `app.py`).

## 🚀 How to Run Locally

### Step 1: Clone the repository

```bash
git clone https://github.com/grvenkat40/Blood-Cell-Classification.git
cd Blood-Cell-Classification
```

### Step 2: Run using Docker

```bash
docker build -t hematovision-app .
docker run -p 5000:5000 hematovision-app
```

Then open: `http://localhost:5000`

---

## 📦 Requirements

For manual installation (if not using Docker):

```txt
flask==2.3.3
tensorflow==2.10.0
opencv-python
gdown
werkzeug==2.3.7
numpy
pandas
```

---

## 📁 Model File

The model `Blood_Cell_PRED.h5` is **automatically downloaded** at runtime using `gdown`. You don’t need to upload it manually.

---

## 📸 Sample Use

Upload a microscopic image of a blood smear → The model predicts the cell type → You get a label and confidence score.

---

## 🔐 Authentication

Includes basic login and registration system using Flask sessions and SQLite for storing user details.

---

## 📈 Results

- ✅ Training Accuracy: **96.6%**
- ✅ Validation Accuracy: **67.8%**
- ✅ Real-time prediction within **2 seconds**

---

## 🧠 Author

**Goondam Ramalingam Venkat**  
📧 grvenkat40@gmail.com  
📍 Siddharth Institute of Engineering & Technology, CSIT

---

## 📜 License

This project is part of an internship initiative by **SmartBridge** and is available for educational and research purposes.

---

## 🌟 Acknowledgements

- [SmartBridge](https://smartbridge.in)
- TensorFlow & Keras Team
- Kaggle (for dataset)
- Flask & Docker Communities
