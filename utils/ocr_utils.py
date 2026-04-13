# utils/ocr_utils.py

import pytesseract
import cv2
import numpy as np
import re
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

def analyze_blood_parameters(ocr_text):
    """
    Analyze blood parameters from OCR text to determine health status
    Handles both numerical values and status indicators (Normal, LOW, HIGH)
    
    Checks:
    - Lymphocyte percentage < 45% OR status "LOW" → unhealthy
    - WBC count > 12,000 OR status "HIGH"/"ELEVATED" → unhealthy
    """
    
    if not ocr_text:
        return {
            "status": "no_data", 
            "confidence": 0.0,
            "parameters": {},
            "health_flags": []
        }
    
    text = ocr_text.lower()
    parameters = {}
    health_flags = []
    abnormal_count = 0
    found_parameters = 0
    
    # ===========================================
    # LYMPHOCYTE Analysis
    # ===========================================
    lymphocyte_status = None
    lymphocyte_value = None
    
    # Method 1: Extract numerical percentage
    lymph_patterns = [
        r'lymphocyte[s]?[:\s]*([0-9]+\.?[0-9]*)\s*%',
        r'lymph[:\s]*([0-9]+\.?[0-9]*)\s*%', 
        r'ly[:\s]*([0-9]+\.?[0-9]*)\s*%'
    ]
    
    for pattern in lymph_patterns:
        match = re.search(pattern, text)
        if match:
            lymphocyte_value = float(match.group(1))
            parameters["lymphocyte_percentage"] = lymphocyte_value
            found_parameters += 1
            # Check threshold: < 45% indicates unhealthy
            if lymphocyte_value < 45:
                health_flags.append("low_lymphocytes")
                abnormal_count += 1
            break
    
    # Method 2: Extract status indicators for lymphocytes
    if lymphocyte_value is None:
        # Look for "Lymphocyte Count × 109/L LOW" format
        lymph_status_patterns = [
            r'lymphocyte[s]?.*?(normal|low|high|elevated)',
            r'lymph.*?(normal|low|high|elevated)'
        ]
        
        for pattern in lymph_status_patterns:
            match = re.search(pattern, text)
            if match:
                lymphocyte_status = match.group(1)
                parameters["lymphocyte_status"] = lymphocyte_status
                found_parameters += 1
                # LOW status indicates unhealthy (< 45% equivalent)
                if lymphocyte_status in ['low']:
                    health_flags.append("low_lymphocytes") 
                    abnormal_count += 1
                break
    
    # ===========================================
    # WBC Analysis  
    # ===========================================
    wbc_status = None
    wbc_value = None
    
    # Method 1: Extract numerical WBC count
    wbc_patterns = [
        r'wbc[:\s]*([0-9,]+)',
        r'white blood cell[s]?[:\s]*([0-9,]+)', 
        r'leukocyte[s]?[:\s]*([0-9,]+)',
        r'w\.?b\.?c[:\s]*([0-9,]+)'
    ]
    
    for pattern in wbc_patterns:
        match = re.search(pattern, text)
        if match:
            # Remove commas and convert to float
            wbc_str = match.group(1).replace(',', '')
            wbc_value = float(wbc_str)
            parameters["wbc_count"] = wbc_value
            found_parameters += 1
            # Check threshold: > 12000 indicates unhealthy
            if wbc_value > 12000:
                health_flags.append("elevated_wbc")
                abnormal_count += 1
            break
    
    # Method 2: Extract status indicators for WBC
    if wbc_value is None:
        # Look for "White Blood Cell Count (WBC) 4 - 11 Normal" format
        wbc_status_patterns = [
            r'white blood cell.*?(normal|low|high|elevated)',
            r'wbc.*?(normal|low|high|elevated)',
            r'leukocyte.*?(normal|low|high|elevated)'
        ]
        
        for pattern in wbc_status_patterns:
            match = re.search(pattern, text)
            if match:
                wbc_status = match.group(1)
                parameters["wbc_status"] = wbc_status
                found_parameters += 1
                # HIGH/ELEVATED status indicates unhealthy (> 12000 equivalent)
                if wbc_status in ['high', 'elevated']:
                    health_flags.append("elevated_wbc")
                    abnormal_count += 1
                break
    
    # ===========================================
    # Overall Health Assessment
    # ===========================================
    if found_parameters == 0:
        return {
            "status": "no_parameters_found",
            "confidence": 0.0,
            "parameters": parameters,
            "health_flags": health_flags,
            "raw_text": ocr_text
        }
    
    # Determine health status
    if abnormal_count == 0:
        status = "healthy"
        confidence = 0.8 if found_parameters >= 2 else 0.6
    else:
        status = "unhealthy" 
        confidence = min(0.7 + (abnormal_count * 0.1), 0.9)
    
    return {
        "status": status,
        "confidence": confidence, 
        "parameters": parameters,
        "health_flags": health_flags,
        "abnormal_indicators": abnormal_count,
        "total_parameters_found": found_parameters,
        "interpretation": {
            "lymphocyte_check": "< 45% indicates immune issues" if "low_lymphocytes" in health_flags else "Within normal range",
            "wbc_check": "> 12000 indicates infection/inflammation" if "elevated_wbc" in health_flags else "Within normal range"
        }
    }
