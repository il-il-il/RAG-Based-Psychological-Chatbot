# download_data.py
import os
import requests
import zipfile

DATA_DIR = "data"
ZIP_PATH = "data.zip"
FILE_ID = "1_UtH9AEcJo4dLMIQVEQ93vT7tYvao-R-"

def download_file_from_google_drive(id, destination):
    URL = "https://drive.google.com/uc?export=download"
    session = requests.Session()
    response = session.get(URL, params={'id': id}, stream=True)
    token = get_confirm_token(response)
    if token:
        response = session.get(URL, params={'id': id, 'confirm': token}, stream=True)
    save_response_content(response, destination)

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value
    return None

def save_response_content(response, destination):
    with open(destination, "wb") as f:
        for chunk in response.iter_content(32768):
            if chunk:
                f.write(chunk)

def ensure_data():
    if not os.path.exists(DATA_DIR):
        print("Downloading data...")
        download_file_from_google_drive(FILE_ID, ZIP_PATH)
        print("Extracting data...")
        with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
            zip_ref.extractall(DATA_DIR)
        os.remove(ZIP_PATH)
    else:
        print("Data folder already exists.")
