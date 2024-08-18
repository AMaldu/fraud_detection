FROM python:3.10-slim

RUN apt-get update && \
    apt-get install -y \
    gcc \
    g++ \
    make \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir pipenv

COPY Pipfile Pipfile.lock /app/
COPY src/ /app/src/
COPY orchestration/models/ /app/orchestration/models/

WORKDIR /app

RUN pipenv install --system --deploy 

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind", "0.0.0.0:9696", "src.predict:app"]

# docker build -t fraud_detection_app -f src/Dockerfile .
#docker run -p 9696:9696 fraud_detection_app
# cd src & python test.py