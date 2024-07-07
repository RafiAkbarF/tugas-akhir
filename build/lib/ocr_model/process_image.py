import os
from ocr_model import ekstrak_informasi_dari_vin_easyocr

def proses_semua_gambar(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.lower().endswith(('.jpeg', '.jpg', '.png')):
            print(f"Memproses {file_path}")
            ekstrak_informasi_dari_vin_easyocr(file_path)

folder_path = "D:/Dataset New"
proses_semua_gambar(folder_path)
