FROM python:3.11-slim
RUN apt-get update && apt-get install -y build-essential wget git && rm -rf /var/lib/apt/lists/*
RUN python -m pip install --upgrade pip
# Install tensorflow matching the development environment
RUN pip install tensorflow==2.21.0

WORKDIR /work

COPY convert.py /work/convert.py
ENTRYPOINT ["python", "/work/convert.py"]
