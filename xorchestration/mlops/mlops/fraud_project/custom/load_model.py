from typing import Dict, List, Tuple

if 'custom' not in globals():
    from mage_ai.data_preparation.decorators import custom

@custom
def models(*args, **kwargs) -> Tuple[List[str], List[Dict[str, str]]]:
    """
    Este bloque devuelve la configuración para RandomForestClassifier.
    """
    model_class: str = 'sklearn.ensemble.RandomForestClassifier'  # Nombre completo de la clase

    child_data: List[str] = [model_class]
    child_metadata: List[Dict] = [dict(block_uuid='RandomForestClassifier')]

    return child_data, child_metadata

