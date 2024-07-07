import os
import cv2
import easyocr
import numpy as np
from matplotlib import pyplot as plt

def tampilkan_gambar(image, title="Gambar"):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis("off")
    plt.show()

def ekstrak_informasi_dari_vin(image_path):
    # Muat gambar menggunakan OpenCV
    image = cv2.imread(image_path)
    tampilkan_gambar(image, title="Gambar Asli")
    
    # Inisialisasi EasyOCR Reader
    reader = easyocr.Reader(['id'])
    
    # Lakukan OCR pada gambar
    results = reader.readtext(image)
    
    # Tampilkan hasil
    print("Teks yang Diekstrak:")
    for (bbox, text, prob) in results:
        print(f"{text} (Kepercayaan: {prob:.2f})")

def ekstrak_informasi_dari_vin_easyocr(image_path):
    # Muat gambar menggunakan OpenCV
    image = cv2.imread(image_path)
    
    # Inisialisasi EasyOCR Reader
    reader = easyocr.Reader(['id'])
    
    # Lakukan OCR pada gambar
    results = reader.readtext(image)
    
    # Tampilkan hasil
    print("Teks yang Diekstrak:")
    for (bbox, text, prob) in results:
        print(f"{text} (Kepercayaan: {prob:.2f})")

def proses_semua_gambar(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.lower().endswith(('.jpeg', '.jpg', '.png')):
            print(f"Memproses {file_path}")
            ekstrak_informasi_dari_vin_easyocr(file_path)

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Model OCR untuk mengekstrak teks dari gambar.')
    parser.add_argument('image_path', type=str, nargs='?', default=None, help='Path ke file gambar')
    parser.add_argument('--method', type=str, choices=['vin', 'easyocr'], default='easyocr',
                        help='Metode untuk OCR (vin atau easyocr)')
    parser.add_argument('--folder', type=str, help='Path ke folder dengan gambar')

    args = parser.parse_args()
    
    if args.folder:
        proses_semua_gambar(args.folder)
    elif args.image_path:
        if args.method == 'vin':
            ekstrak_informasi_dari_vin(args.image_path)
        else:
            ekstrak_informasi_dari_vin_easyocr(args.image_path)
    else:
        print("Anda harus memberikan path ke gambar atau folder yang berisi gambar.")

if __name__ == "__main__":
    main()
