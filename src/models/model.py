import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import pickle 

# Cargar los datos
df = pd.read_parquet('../../data/processed/df_final.parquet', engine='fastparquet')

# Definir columnas categóricas y numéricas
numerical_cols = df.select_dtypes(include=['float32', 'int16']).columns
categorical_cols = df.select_dtypes(include=['object', 'category']).columns

# Definir el transformador personalizado para imputar outliers
class OutlierImputer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        self.columns_ = X.columns
        return self

    def transform(self, X):
        X = X.copy()
        for column in X.columns:
            Q1 = X[column].quantile(0.25)
            Q3 = X[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            X[column] = np.where((X[column] < lower_bound) | (X[column] > upper_bound),
                                 X[column].mean(), X[column])
        return X

# Crear el preprocesador
preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('outlier_imputer', OutlierImputer()),
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', StandardScaler())
        ]), numerical_cols),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ]), categorical_cols)
    ])

# Asegúrate de usar el nombre correcto para la columna objetivo
target_col = 'isFraud'  # Columna objetivo actualizada

# Separar las características y la variable objetivo
X = df.drop(target_col, axis=1)
y = df[target_col]

# Asegúrate de que la variable objetivo es categórica
unique_values = y.unique()
if len(unique_values) != 2:
    # Si hay más de dos valores únicos, convierte a binario
    y = pd.Categorical(y).codes
else:
    # Si es continua pero tiene solo dos valores, convierte a 0 y 1
    y = y.map({unique_values[0]: 0, unique_values[1]: 1})

# Dividir los datos en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Crear el pipeline con solo el modelo Random Forest
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Entrenar y evaluar el modelo Random Forest
with mlflow.start_run(run_name='Random Forest'):
    print("\nTraining Random Forest...")
    
    # Ajustar el modelo
    pipeline.fit(X_train, y_train)
    
    # Evaluar el modelo
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:, 1]  # Probabilidades de la clase positiva
    
    # Calcular métricas
    classification_rep = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    fpr, tpr, _ = roc_curve(y_test, y_prob, pos_label=1)  # Asegúrate de pasar el pos_label
    
    # Imprimir y guardar resultados
    print(f"\nRandom Forest Classification Report:")
    print(classification_rep)
    
    print(f"Random Forest Confusion Matrix:")
    print(conf_matrix)
    
    print(f"Random Forest ROC AUC Score: {roc_auc:.2f}")

    # Log metrics
    mlflow.log_metric("roc_auc_score", roc_auc)
    mlflow.log_params(pipeline.named_steps['classifier'].get_params())
    
    # Log model
    mlflow.sklearn.log_model(pipeline, "model")
    
    # Save the ROC curve plot
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label=f'Random Forest ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (Random Forest)')
    plt.legend(loc="lower right")
    plt.savefig('Random_Forest_roc_curve.png')
    mlflow.log_artifact('Random_Forest_roc_curve.png')


with open('random_forest_pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline, f)