import pandas as pd
import category_encoders as ce
from typing import Optional

def select_features(df: pd.DataFrame, target: str, pickle_filepath: Optional[str] = None) -> pd.DataFrame:
    # Seleccionar columnas numéricas y categóricas
    numerical_cols = df.select_dtypes(include=['float32', 'int16', 'int8', 'int64', 'float64']).columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns

    # Verificar si la columna objetivo está en las columnas numéricas
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame columns.")
    
    if target in numerical_cols:
        numerical_cols = numerical_cols[numerical_cols != target]
    
    # Aplicar codificación
    encoder = ce.TargetEncoder(cols=categorical_cols)
    df_encoded = encoder.fit_transform(df, df[target])

    # Calcular matriz de correlación
    correlation_matrix = df_encoded.corr()

    # Identificar pares de columnas con alta correlación
    threshold = 0.9
    high_correlation_pairs = [(col1, col2) for col1 in correlation_matrix.columns for col2 in correlation_matrix.columns 
                              if col1 != col2 and abs(correlation_matrix.loc[col1, col2]) > threshold]

    # Identificar columnas para eliminar
    variables_to_remove = set()
    for col1, col2 in high_correlation_pairs:
        if col1 not in variables_to_remove and col2 not in variables_to_remove:
            variables_to_remove.add(col2)
    


    # Crear DataFrame reducido
    df = df_encoded.drop(columns=variables_to_remove)

    
    return df
