# utils/ocr_utils.py

import pytesseract
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)

def extract_text(file):
    """
    file: PIL Image or bytes
    """
    # Convert bytes to PIL Image
    if isinstance(file, bytes):
        file = Image.open(BytesIO(file)).convert("RGB")
    
    # Convert PIL Image to numpy array for OpenCV
    img = np.array(file)
    if img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Run OCR
    text = pytesseract.image_to_string(gray)
    
    return text
