# services/cattle_disease_service.py

import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
from io import BytesIO

from fastapi import UploadFile

from utils.ocr_utils import extract_text
from utils.symptom_utils import analyze_symptoms
from utils.fusion_utils import fuse_results

# ------------------------------
# MODEL CONFIG
# ------------------------------
MODEL_PATH = "Model/cattle_multimodal/image_model.pth"
CLASSES = ["Healthy", "FMD", "LSD"]

model = models.vgg16(weights=None)
model.classifier[6] = torch.nn.Linear(4096, len(CLASSES))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# ------------------------------
# IMAGE TRANSFORM
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# ------------------------------
# MAIN PREDICTION FUNCTION
# ------------------------------
async def run_prediction(
    image_file: UploadFile,
    report_file: UploadFile,
    symptoms_text: str
):
    result = {}

    # --------------------------
    # IMAGE PREDICTION
    # --------------------------
    if image_file:
        try:
            img = Image.open(image_file.file).convert("RGB")
            img = transform(img).unsqueeze(0)

            with torch.no_grad():
                outputs = model(img)
                probabilities = F.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)

            result["image_prediction"] = CLASSES[predicted.item()]
            result["image_confidence"] = float(confidence.item())

        except Exception as e:
            result["image_error"] = str(e)

    # --------------------------
    # REPORT (OCR)
    # --------------------------
    if report_file:
        try:
            report_bytes = await report_file.read()
            img = Image.open(BytesIO(report_bytes)).convert("RGB")  # Convert to PIL Image
            text = extract_text(img)  
            result["blood_report"] = text

        except Exception as e:
            result["report_error"] = str(e)

    # --------------------------
    # SYMPTOMS ANALYSIS
    # --------------------------
    if symptoms_text:
        try:
            symptoms_result = analyze_symptoms(symptoms_text)
            result["symptoms_analysis"] = symptoms_result["scores"]
            result["symptom_confidence"] = symptoms_result["confidence"]
        except Exception as e:
            result["symptoms_error"] = str(e)

    # --------------------------
    # FUSION LOGIC
    # --------------------------
    try:
        fusion_result = fuse_results(result)
        result["final_decision"] = fusion_result["prediction"]
        result["overall_confidence"] = fusion_result["confidence"]
        result["severity_assessment"] = fusion_result["severity"]
    except Exception as e:
        result["fusion_error"] = str(e)

    return result
