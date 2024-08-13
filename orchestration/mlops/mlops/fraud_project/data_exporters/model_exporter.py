from typing import Dict, Tuple
from sklearn.base import BaseEstimator
import joblib
from pathlib import Path



if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

@data_exporter
def model_exporter(
    model_and_info: Tuple[BaseEstimator, Dict[str, str]],  # Tupla con el modelo y la información del modelo
    **kwargs,
) -> None:
    """
    Función para exportar el modelo y la información del modelo a un archivo.
    
    :param model_and_info: Tupla con el modelo y la información del modelo.
    :param kwargs: Argumentos adicionales.
    """
    # Los valores de la tupla son ya el modelo y la información del modelo
    model, model_info = model_and_info
    
    # Imprimir para depuración
    print("Modelo recibido:", model)
    print("Información del modelo:", model_info)
    
    # Extraer el nombre del archivo desde model_info
    model_filename = model_info.get('filename', 'random_forest.pkl')  # Nombre de archivo por defecto
    
    # Crear la ruta para guardar el modelo
    models_dir = Path('/root/src/fraud_project/final_model')
    models_dir.mkdir(parents=True, exist_ok=True)  # Asegura que el directorio existe
    
    # Ruta completa del archivo donde se guardará el modelo
    model_path = models_dir / model_filename
    
    # Guardar el modelo utilizando joblib
    try:
        joblib.dump(model, model_path)
        print(f"Modelo guardado en: {model_path}")
    except Exception as e:
        print(f"Error al guardar el modelo: {e}")

