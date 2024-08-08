
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from hyperopt import hp, fmin, tpe, Trials

# Configuración de MLflow
mlflow.set_experiment('fraud_detection_experiment')

# Cargar el DataFrame procesado
X_train = pd.read_csv('../data/gold/X_train_scaled.csv')
X_test = pd.read_csv('../data/gold/X_test_scaled.csv')
y_train = pd.read_csv('../data/gold/y_train.csv')
y_test = pd.read_csv('../data/gold/y_test.csv')

# Definir la función objetivo para hyperopt
def objective(params):
    # Crear el modelo con los parámetros de hyperopt
    model = RandomForestClassifier(
        n_estimators=int(params['n_estimators']),
        max_depth=int(params['max_depth']),
        min_samples_split=int(params['min_samples_split']),
        random_state=42
    )
    
    # Entrenar el modelo
    model.fit(X_train, y_train.values.ravel())
    
    # Hacer predicciones
    y_pred = model.predict(X_test)
    
    # Calcular la precisión como la métrica de evaluación
    accuracy = (confusion_matrix(y_test, y_pred)[0,0] + confusion_matrix(y_test, y_pred)[1,1]) / confusion_matrix(y_test, y_pred).sum()
    
    return -accuracy  # hyperopt busca minimizar la función objetivo, por lo que retornamos el negativo de la precisión

# Definir el espacio de búsqueda de hiperparámetros
space = {
    'n_estimators': hp.quniform('n_estimators', 50, 200, 10),
    'max_depth': hp.quniform('max_depth', 10, 50, 5),
    'min_samples_split': hp.quniform('min_samples_split', 2, 10, 1)
}

# Iniciar un nuevo experimento
with mlflow.start_run():
    
    # Usar hyperopt para encontrar los mejores hiperparámetros
    trials = Trials()
    best = fmin(
        fn=objective,
        space=space,
        algo=tpe.suggest,
        max_evals=20,  # Número de iteraciones de búsqueda
        trials=trials
    )
    
    # Imprimir los mejores hiperparámetros encontrados
    print("Mejores hiperparámetros encontrados:")
    print(best)
    
    # Crear el modelo con los mejores hiperparámetros
    best_model = RandomForestClassifier(
        n_estimators=int(best['n_estimators']),
        max_depth=int(best['max_depth']),
        min_samples_split=int(best['min_samples_split']),
        random_state=42
    )
    
    # Entrenar el modelo
    best_model.fit(X_train, y_train.values.ravel())
    
    # Hacer predicciones
    y_pred = best_model.predict(X_test)
    
    # Evaluar el modelo
    confusion = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    
    print("Confusion Matrix:")
    print(confusion)
    print("\nClassification Report:")
    print(classification_rep)
    
    # Registrar métricas
    mlflow.log_metric("accuracy", (confusion[0,0] + confusion[1,1]) / confusion.sum())
    
    # Registrar el modelo
    mlflow.sklearn.log_model(best_model, "random_forest_model")
    
    # Guardar el modelo localmente como respaldo
    joblib.dump(best_model, '../models/random_forest_model.pkl')
    print("Modelo guardado localmente en '../models/random_forest_model.pkl'")

print("Entrenamiento, ajuste de hiperparámetros y registro de modelo completado con MLflow.")

