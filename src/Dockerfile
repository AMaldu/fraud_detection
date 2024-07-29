FROM python:3.10.12-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app


COPY ['Pipfile', 'Pipfile.lock', "./"] 
# "./" = current directory


RUN pipenv install --system --deploy

COPY ['predict.py', 'pipeline_model.pkl', './']

#Expose the port 9696 to be able to listen
EXPOSE 9696 

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0.:9696", "predict:app" ]