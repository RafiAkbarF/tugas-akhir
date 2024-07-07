import cv2
import easyocr
import matplotlib.pyplot as plt


# Fungsi untuk menampilkan gambar
def display_img(img, title='Gambar'):
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

# Fungsi untuk mempreproses gambar
def preprocess_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(gray)
    
    # Otsu thresholding
    _, binary = cv2.threshold(cl1, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary

# Fungsi untuk mengekstrak teks menggunakan metode custom
def extract_information_from_vin(image_path):
    image = preprocess_image(image_path)
    display_img(image, 'Gambar yang Dipreproses')
    reader = easyocr.Reader(['id'])
    results = reader.readtext(image)
    print("Teks yang Diekstrak:")
    combined_text = " ".join([text for (_, text, _) in results])
    for (bbox, text, prob) in results:
        print(f"{text} (Kepercayaan: {prob:.2f})")
    print(combined_text)
if __name__ == "__main__":
    image_path = "D:/Dataset_New/2.jpg"
    image = preprocess_image(image_path)
    if image is None:
        print("Gambar tidak ditemukan atau tidak bisa dibaca.")
    else:
        print("Gambar berhasil dibaca.")
    extract_information_from_vin(image_path)
