import requests

from predict import predict, prepare_features

customer = {
    "step": 5,
    "type": "CASH_OUT",
    "amount": 23453.6,
    "nameOrig": "C1231006815",
    "oldbalanceOrig": 181.0,
    "newbalanceOrig": 160296.359375,
    "nameDest": "M1979787155",
    "oldbalanceDest": 21182.00000,
    "newbalanceDest": 214661.4375,
}
customer_df = prepare_features(customer)


pred = predict(customer_df)
print(pred[0])


url = "http://localhost:9696/predict"

response = requests.post(url, json=customer)
print(response.json())
