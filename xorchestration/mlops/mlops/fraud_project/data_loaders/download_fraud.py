import requests
import os

if 'data_loader' not in globals():
    from mage_ai.data_preparation.decorators import data_loader
if 'test' not in globals():
    from mage_ai.data_preparation.decorators import test


@data_loader
def download_file(url: str, local_path: str) -> None:
    """
    Descarga un archivo desde una URL y lo guarda en una ruta local.
    """
    # Realizar la solicitud HTTP para obtener el archivo
    response = requests.get(url, stream=True)
    
    if response.status_code != 200:
        raise Exception(f"Failed to download file: {response.status_code} {response.reason}")

    # Crear el directorio si no existe
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    
    # Guardar el archivo
    with open(local_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

@test
def test_download_file() -> None:
    """
    Test function to validate file download.
    """
    url = 'https://media.githubusercontent.com/media/AMaldu/fraud_detection/main/data/bronze/PS_20174392719_1491204439457_log.csv'
    local_path = 'data/PS_20174392719_1491204439457_log.csv'
    download_file(url, local_path)

    # Verify that the file has been downloaded
    if not os.path.isfile(local_path):
        raise Exception(f"File not found after download: {local_path}")

    print(f"File downloaded successfully and verified at: {local_path}")