import torch
from torchvision import models, transforms
from PIL import Image

from utils.ocr_utils import extract_text
from utils.symptom_utils import analyze_symptoms
from utils.fusion_utils import fuse_results

MODEL_PATH = "Model/cattle_multimodal/image_model.pth"
CLASSES = ["Healthy", "FMD", "LSD"]

model = models.vgg16()
model.classifier[6] = torch.nn.Linear(4096, 3)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225])
])

def run_prediction(image_file, report_file, symptoms_text):
    result = {}

    # IMAGE
    if image_file:
        img = Image.open(image_file).convert("RGB")
        img = transform(img).unsqueeze(0)
        with torch.no_grad():
            out = model(img)
            _, pred = out.max(1)
        result["image_prediction"] = CLASSES[pred.item()]

    # REPORT
    if report_file:
        text = extract_text(report_file)
        result["blood_report"] = text

    # SYMPTOMS
    if symptoms_text:
        result["symptoms"] = analyze_symptoms(symptoms_text)

    # FUSION
    result["final_decision"] = fuse_results(result)

    return result
