import pytesseract
import cv2
import tempfile

def extract_text(file):
    with tempfile.NamedTemporaryFile(delete=False) as temp:
        file.save(temp.name)
        img = cv2.imread(temp.name)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return pytesseract.image_to_string(gray)
