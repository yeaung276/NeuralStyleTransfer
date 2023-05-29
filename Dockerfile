FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10

COPY requirements.txt /app/

RUN pip install -r /app/requirements.txt

RUN wget https://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat -P /app

COPY . /app

CMD ["python3", "/app/app.py"]