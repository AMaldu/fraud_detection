from typing import Tuple
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split


def preprocessor(
    df: pd.DataFrame, target: str, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    print(df)
    categorical_features = ['type', 'nameDest']
    numeric_features = ['step', 'amount', 'oldbalanceOrig', 'oldbalanceDest', 'diffbalanceOrig', 'diffbalanceDest']
    
    # Definir transformadores
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')
    numeric_transformer = StandardScaler()
    
    # Crear el preprocesador
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ]
    )
    
    # Crear el pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor)
    ])
    
    # Separar características y objetivo
    X = df.drop(columns=target)
    y = df[target]
    
    # Dividir los datos en entrenamiento y prueba antes de aplicar el preprocesamiento
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state
    )
    
    # Ajustar y transformar los datos de entrenamiento
    X_train_preprocessed = pipeline.fit_transform(X_train)
    # Transformar los datos de prueba
    X_test_preprocessed = pipeline.transform(X_test)
    
    return X_train_preprocessed, X_test_preprocessed, y_train, y_test