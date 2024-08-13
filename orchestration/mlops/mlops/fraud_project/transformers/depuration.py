from typing import Any

if 'transformer' not in globals():
    from mage_ai.data_preparation.decorators import transformer

@transformer
def debug_training_set(training_set_data: Any, *args, **kwargs) -> None:
    """
    Este bloque imprime y verifica los datos recibidos del Global Data Product
    del conjunto de entrenamiento.
    """
    # Imprimir el tipo de datos recibidos
    print(f"Tipo de datos recibidos: {type(training_set_data)}")
    
    if isinstance(training_set_data, dict):
        print(f"Número de claves en el diccionario: {len(training_set_data)}")
        # Verificar la clave "splitting_data"
        if 'splitting_data' in training_set_data:
            splitting_data = training_set_data['splitting_data']
            print(f"Tipo de splitting_data: {type(splitting_data)}")
            if isinstance(splitting_data, list):
                print(f"Número de elementos en splitting_data: {len(splitting_data)}")
                # Inspeccionar cada elemento en la lista
                for i, element in enumerate(splitting_data):
                    print(f"Elemento {i}: Tipo: {type(element)}, Tamaño: {getattr(element, 'shape', 'N/A')}")
            else:
                print("splitting_data no es una lista.")
        else:
            print("La clave 'splitting_data' no está presente en el diccionario.")
    else:
        print("Datos recibidos no están en el formato esperado de diccionario.")