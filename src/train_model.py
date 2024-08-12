
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from hyperopt import hp, fmin, tpe, Trials

mlflow.set_experiment('fraud_detection_experiment')

X_train = pd.read_csv('../data/gold/X_train_scaled.csv')
X_test = pd.read_csv('../data/gold/X_test_scaled.csv')
y_train = pd.read_csv('../data/gold/y_train.csv')
y_test = pd.read_csv('../data/gold/y_test.csv')

def objective(params):
    model = RandomForestClassifier(
        n_estimators=int(params['n_estimators']),
        max_depth=int(params['max_depth']),
        min_samples_split=int(params['min_samples_split']),
        random_state=42
    )
    
    model.fit(X_train, y_train.values.ravel())
    
    y_pred = model.predict(X_test)
    
    accuracy = (confusion_matrix(y_test, y_pred)[0,0] + confusion_matrix(y_test, y_pred)[1,1]) / confusion_matrix(y_test, y_pred).sum()
    
    return -accuracy  

space = {
    'n_estimators': hp.quniform('n_estimators', 50, 200, 10),
    'max_depth': hp.quniform('max_depth', 10, 50, 5),
    'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1)
}

with mlflow.start_run():
    
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=20,  
    )
    
    print("Mejores hiperparámetros encontrados:")
    print(best)
    
    best_model = RandomForestClassifier(
        n_estimators=int(best['n_estimators']),
        max_depth=int(best['max_depth']),
        min_samples_split=int(best['min_samples_split']),
        random_state=42
    )
    
    best_model.fit(X_train, y_train.values.ravel())
    
    y_pred = best_model.predict(X_test)
    
    confusion = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    
    print("Confusion Matrix:")
    print(confusion)
    print("\nClassification Report:")
    print(classification_rep)
    
    mlflow.log_metric("accuracy", (confusion[0,0] + confusion[1,1]) / confusion.sum())
    
    mlflow.sklearn.log_model(best_model, "random_forest_model")
    
    joblib.dump(best_model, '../models/random_forest_model.pkl')
    print("Modelo guardado localmente en '../models/random_forest_model.pkl'")

print("Entrenamiento, ajuste de hiperparámetros y registro de modelo completado con MLflow.")

