# check.py
import os
import gdown

model_path = 'dog_cat_model.h5'
gdrive_file_id = '1W3o5hvlGU7I0gpDdtKgfctsscZKiXuya'
gdown_url = f'https://drive.google.com/uc?id={gdrive_file_id}'

if not os.path.exists(model_path):
    print("Downloading model...")
    gdown.download(gdown_url, model_path, quiet=False)
else:
    print("Model already exists.")
