
import pickle
from typing import Any
from flask import Flask, request, jsonify

def load_and_predict_model(X_test: Any, filename='pipeline_model.pkl'):
    with open(filename, 'rb') as file:
        loaded_pipeline = pickle.load(file)
    y_pred_loaded = loaded_pipeline.predict(X_test)    
    return y_pred_loaded


app = Flask('fraud-prediction')


@app.route('/predict', methods = ['POST'])
def predict_endpoit():
    customer = request.get_json()
    pred = load_and_predict_model(customer)
    
    result = {
        'isFraud':pred
    }

    return jsonify(result)




if __name__ == "__main__":
    app.run(debug = True, host = '0.0.0.0', port = 9696)