from pathlib import Path
import joblib

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

@transformer
def model_exporter(model_and_info, **kwargs):
    model, model_info = model_and_info

    print("Modelo recibido:", model)
    print("Información del modelo:", model_info)
    
    model_filename = model_info.get('filename', 'random_forest.pkl')
    models_dir = Path('/root/fraud_project/final_model')
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / model_filename
    try:
        joblib.dump(model, model_path)
        print(f"Modelo guardado en: {model_path}")
    except Exception as e:
        print(f"Error al guardar el modelo: {e}")
