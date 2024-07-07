import torch
import cv2
import matplotlib.pyplot as plt
import easyocr
from pathlib import Path

# Function to load YOLOv5 model correctly
def load_model(path):
    return torch.hub.load('ultralytics/yolov5', 'custom', path=path, force_reload=True)

# Path to the custom YOLOv5 model
model_path = r'D:\Tugas Akhir\TrainingData\ocr_model_project\best.pt'

# Load the custom YOLOv5 model
model = load_model(model_path)

# Path to the image
img_path = r'D:\Dataset New\2.1.jpeg'

# Load image
img = cv2.imread(img_path)

# Check if image is loaded correctly
if img is None:
    raise ValueError(f"Image not loaded correctly. Check the path: {img_path}")

# Perform inference
results = model(img)

# Convert results to pandas dataframe
df = results.pandas().xyxy[0]

# Initialize EasyOCR reader
reader = easyocr.Reader(['en'])

# Extract and print text, and draw bounding boxes
for _, row in df.iterrows():
    x1, y1, x2, y2 = map(int, row[['xmin', 'ymin', 'xmax', 'ymax']])
    cropped_img = img[y1:y2, x1:x2]
    result = reader.readtext(cropped_img)
    # Draw bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    # Display the detected text
    if result:
        for (bbox, text, prob) in result:
            cv2.putText(img, f'{text} ({prob:.2f})', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
            print(f'Detected text: {text} (Confidence: {prob:.2f})')

# Convert BGR to RGB for plotting with matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Plot the image with bounding boxes and text
plt.figure(figsize=(12, 8))
plt.imshow(img_rgb)
plt.axis('off')
plt.show()
