FROM python:3.9-slim

WORKDIR /app

# Install system-level OpenCV dependency
RUN apt-get update && apt-get install -y libgl1

COPY . /app

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Download model using gdown
RUN python -c "import os, gdown; \
url = 'https://drive.google.com/uc?id=12Bbh3kaEBFsE2WLr3ymufSdu4bfi1WUr'; \
output = 'Blood_Cell_PRED.h5'; \
gdown.download(url, output, quiet=False) if not os.path.exists(output) else print('Model exists')"

EXPOSE 5000

CMD ["python", "app.py"]
