import os
import pickle
from typing import Any
from flask import Flask, request, jsonify
import pandas as pd


# run gunicorn --bind=0.0.0.0:9696 src.predict:app


def load_and_predict_model():
    base_path = os.path.dirname(__file__)  # Obtiene el directorio actual
    model_path = os.path.join(base_path, '../orchestration/models/random_forest_classifier.b')
    prep_path = os.path.join(base_path, '../orchestration/models/preprocessor_pipeline.b')
    
    # Carga el modelo
    with open(model_path, "rb") as f_in:
        model = pickle.load(f_in)

    # Carga el preprocesador
    with open(prep_path, "rb") as f_in:
        preprocessor = pickle.load(f_in)

    return preprocessor, model

preprocessor, model = load_and_predict_model()

def prepare_features(customer: dict) -> pd.DataFrame:
    features = {
        'step': customer['step'],
        'type': customer['type'],
        'amount': customer['amount'],
        'nameOrig': customer['nameOrig'],
        'oldbalanceOrig': customer['oldbalanceOrig'],
        'newbalanceOrig': customer['newbalanceOrig'],
        'nameDest': customer['nameDest'],
        'oldbalanceDest': customer['oldbalanceDest'],
        'newbalanceDest': customer['newbalanceDest'],
        'diffbalanceOrig': customer['newbalanceOrig'] - customer['oldbalanceOrig'],
        'diffbalanceDest': customer['newbalanceDest'] - customer['oldbalanceDest']
    }
    
    return pd.DataFrame([features])

def predict(features: pd.DataFrame) -> Any:
    X = preprocessor.transform(features)
    preds = model.predict(X)
    return preds

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    customer = request.get_json()
    features_df = prepare_features(customer)
    pred = predict(features_df)
    
    result = {
        'isFraud': int(pred[0])
    }

    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)

