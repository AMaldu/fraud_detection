import requests

customer = {
    'step': 5,
    'type': CASH_OUT,
    'amount': 23453.6,
	'oldbalanceOrig': 0.0,
    'newbalanceOrig': 181.0,
    'nameDest': 235.860018,
    'oldbalanceDest': 21182.00000,
    'newbalanceDest': 214661.4375
    
}

url = 'http://localhost:9696/predict'

response = requests.post(url, json=customer)
print(response.json())