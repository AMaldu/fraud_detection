import os
import pandas as pd
from prefect import task, Flow
from datetime import timedelta
from pandas import DataFrame
import category_encoders as ce
from typing import Tuple
from scipy.sparse import csr_matrix
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score

@task
def ingest_files() -> pd.DataFrame:
    """Read data into dataframe"""
    file_path = 'data/bronze/PS_20174392719_1491204439457_log.csv'
    
    df = pd.read_csv(file_path, nrows = 1000)
    
    return df

@task
def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Renaming columns
    df = df.rename(columns={'oldbalanceOrg': 'oldbalanceOrig'})

    # Transform type to category to optimize memory and speed
    df['step'] = df['step'].astype('int16')
    df[['type', 'nameOrig', 'nameDest']] = df[['type', 'nameOrig', 'nameDest']].astype('category')
    df[['amount', 'oldbalanceOrig', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFraud']] = df[['amount', 'oldbalanceOrig', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest', 'isFraud']].astype('float32')

    print("Columnas después de limpiar:", df.columns)
    return df

@task
def combine_features(df: DataFrame) -> DataFrame:
    # Calculating balance differences
    df['diffbalanceOrig'] = df['newbalanceOrig'] - df['oldbalanceOrig']
    df['diffbalanceDest'] = df['newbalanceDest'] - df['oldbalanceDest']

    # Dropping unnecessary columns
    df = df.drop(columns=['isFlaggedFraud'])
    print("Columnas después de combinar características:", df.columns)
    return df

@task
def select_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    # Identifica columnas numéricas y categóricas
    numerical_cols = df.select_dtypes(include=['float32', 'int16', 'int8', 'int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    # Verifica si la columna target está en el DataFrame
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame columns.")
    
    # Si target está en columnas numéricas, lo excluye de la lista de columnas numéricas
    if target in numerical_cols:
        numerical_cols = numerical_cols[numerical_cols != target]
    
    # Codifica variables categóricas usando TargetEncoder
    encoder = ce.TargetEncoder(cols=categorical_cols)
    df_encoded = encoder.fit_transform(df, df[target])

    # Calcula la matriz de correlación
    correlation_matrix = df_encoded.corr()

    # Identifica pares de columnas con alta correlación
    threshold = 0.9
    high_correlation_pairs = [(col1, col2) for col1 in correlation_matrix.columns for col2 in correlation_matrix.columns 
                              if col1 != col2 and abs(correlation_matrix.loc[col1, col2]) > threshold]

    # Establece las variables a eliminar, pero siempre conserva la columna target
    variables_to_remove = set()
    for col1, col2 in high_correlation_pairs:
        if col1 != target and col2 != target:
            if col1 not in variables_to_remove and col2 not in variables_to_remove:
                variables_to_remove.add(col2)

    # Elimina columnas altamente correlacionadas, pero conserva la columna target
    df = df_encoded.drop(columns=variables_to_remove, errors='ignore')
    
    print("Columnas después de seleccionar características:", df.columns)
    
    return df

@task
def preprocessor(
    df: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = 42
) -> Tuple[csr_matrix, csr_matrix, pd.Series, pd.Series]:
    """Preprocess data and split into train/test sets"""
    categorical_features = ['type']
    numeric_features = ['step', 'amount', 'oldbalanceOrig', 'oldbalanceDest', 'diffbalanceOrig', 'diffbalanceDest']
    
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
    numeric_transformer = StandardScaler()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])
    
    X = df.drop(columns=target)
    y = df[target]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )
    
    X_train = pipeline.fit_transform(X_train)
    X_test = pipeline.transform(X_test)
    
    pipeline_path = "orchestration/models/preprocessor_pipeline.b"
    with open(pipeline_path, "wb") as f_out:
        pickle.dump(pipeline, f_out)
    
    print(f"Pipeline guardado en '{pipeline_path}'.")
    return X_train, X_test, y_train, y_test



@task(log_prints=True)
def train_best_model(X_train: csr_matrix, X_test: csr_matrix, y_train: pd.Series, y_test: pd.Series):
    model = RandomForestClassifier(
        n_estimators=190,
        max_depth=45,
        min_samples_split=7,
        random_state=42
    )
    
    # Entrenar el modelo
    model.fit(X_train, y_train)
    
    # Realizar predicciones en el conjunto de validación
    y_pred = model.predict(X_test)
    
    # Calcular las métricas de evaluación
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    # Imprimir las métricas (o puedes loguearlas en otro lugar si lo prefieres)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    
    # Guardar el modelo en la carpeta especificada
    os.makedirs("orchestration/models", exist_ok=True)
    with open("orchestration/models/random_forest_classifier.b", "wb") as f_out:
        pickle.dump(model, f_out)
    
    print("Modelo guardado en 'orchestration/models/random_forest_classifier.b'.")

    
    return None
        



@Flow
def main_flow():
    df = ingest_files()
    df_cleaned = clean(df)
    df_combined = combine_features(df_cleaned)
    df_selected = select_features(df_combined, target='isFraud')
    print("Columnas de df_selected:", df_selected.columns)
    X_train, X_test, y_train, y_test = preprocessor(df_selected, target='isFraud')
    train_best_model(X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    main_flow()
