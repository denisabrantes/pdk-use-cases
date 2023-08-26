FROM python:3.8

WORKDIR /app

COPY requirements.txt /app
RUN pip install -r requirements.txt --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org --trusted-host pypi.nvidia.com

COPY deploy.py /app
COPY common.py /app
COPY object_detection_handler.py /app

ENV PYTHONUNBUFFERED=1
