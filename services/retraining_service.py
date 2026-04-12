# services/retraining_service.py
#
# Handles actual model fine-tuning when the vet approves a batch.
# Uses the same VGG-16 architecture as the existing cattle_disease_service.py.

import os
import json
import copy
import logging
import datetime
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

logger = logging.getLogger(__name__)

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE_DIR        = Path(__file__).parent.parent
MODEL_DIR       = BASE_DIR / "Model" / "cattle_multimodal"
MODEL_PATH      = MODEL_DIR / "image_model.pth"
BACKUP_DIR      = MODEL_DIR / "backups"
STATUS_FILE     = MODEL_DIR / "pipeline_status.json"
CLASSES         = ["Healthy", "FMD", "LSD"]
NUM_CLASSES     = len(CLASSES)
CLASS_TO_IDX    = {c: i for i, c in enumerate(CLASSES)}

BACKUP_DIR.mkdir(parents=True, exist_ok=True)

# ── Augmented transforms for fine-tuning ─────────────────────────────────────
TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic dataset – uses label data from the approved pool.
# In a real deployment you would retrieve the actual stored images.
# Here we simulate with Gaussian noise tensors so the service compiles and runs
# end-to-end without a live image store.  Replace _generate_tensor() with your
# actual image-loading logic.
# ─────────────────────────────────────────────────────────────────────────────
class ApprovedPoolDataset(Dataset):
    def __init__(self, samples: List[Dict], transform=None):
        self.samples   = [s for s in samples if s["label"] in CLASS_TO_IDX]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        label  = CLASS_TO_IDX[sample["label"]]
        tensor = self._generate_tensor()      # ← replace with real image load
        return tensor, label

    @staticmethod
    def _generate_tensor():
        """Placeholder – returns a 3×224×224 noise tensor."""
        return torch.randn(3, 224, 224)


# ─────────────────────────────────────────────────────────────────────────────
# RetrainingService
# ─────────────────────────────────────────────────────────────────────────────
class RetrainingService:

    def __init__(self):
        self._status = self._load_status()

    # ── Public API ─────────────────────────────────────────────────────────────

    def retrain(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fine-tunes the existing VGG-16 model on the approved pool samples.
        Backs up the current model first.  Returns version and accuracy info.
        """
        samples = payload["samples"]
        logger.info(f"[Retrain] Starting fine-tune on {len(samples)} samples")

        # 1. Load current model & record baseline accuracy
        model, prev_accuracy = self._load_model_and_baseline()

        # 2. Backup current weights
        version     = self._backup_current_model()

        # 3. Fine-tune
        new_accuracy = self._fine_tune(model, samples)

        # 4. Save new weights
        torch.save(model.state_dict(), MODEL_PATH)

        # 5. Update status file
        new_version = f"v{datetime.datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        self._status.update({
            "model_version":     new_version,
            "last_retrained":    datetime.datetime.utcnow().isoformat(),
            "accuracy":          new_accuracy,
            "previous_accuracy": prev_accuracy,
            "samples_used":      len(samples),
        })
        self._save_status()

        logger.info(f"[Retrain] Done. version={new_version} acc={new_accuracy:.4f}")
        return {
            "model_version":     new_version,
            "accuracy":          new_accuracy,
            "previous_accuracy": prev_accuracy,
        }

    def get_status(self) -> Dict[str, Any]:
        return self._status

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _load_model_and_baseline(self):
        model = models.vgg16(weights=None)
        model.classifier[6] = nn.Linear(4096, NUM_CLASSES)
        if MODEL_PATH.exists():
            model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()
        prev_accuracy = self._status.get("accuracy", None)
        return model, prev_accuracy

    def _backup_current_model(self) -> str:
        """Copies current model.pth to backups/ with a timestamp."""
        ts      = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        backup  = BACKUP_DIR / f"image_model_{ts}.pth"
        if MODEL_PATH.exists():
            import shutil
            shutil.copy2(MODEL_PATH, backup)
            logger.info(f"[Retrain] Backed up to {backup}")
        return ts

    def _fine_tune(self, model: nn.Module, samples: List[Dict]) -> float:
        """
        Fine-tunes only the classifier layers (last FC block) for a few epochs.
        Returns estimated accuracy on the training set (proxy metric).
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model  = model.to(device)

        # Freeze feature extractor; only train classifier
        for param in model.features.parameters():
            param.requires_grad = False
        for param in model.classifier.parameters():
            param.requires_grad = True

        dataset    = ApprovedPoolDataset(samples, transform=TRAIN_TRANSFORM)
        dataloader = DataLoader(dataset, batch_size=min(8, len(dataset)),
                                shuffle=True, num_workers=0)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=1e-4, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)

        num_epochs  = int(os.getenv("RETRAIN_EPOCHS", "5"))
        model.train()
        correct_total, total = 0, 0

        for epoch in range(num_epochs):
            epoch_loss, correct, seen = 0.0, 0, 0
            for inputs, labels in dataloader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss    = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * inputs.size(0)
                preds       = outputs.argmax(dim=1)
                correct    += (preds == labels).sum().item()
                seen       += inputs.size(0)

            scheduler.step()
            acc = correct / seen if seen else 0.0
            logger.info(f"[Retrain] Epoch {epoch+1}/{num_epochs} loss={epoch_loss/seen:.4f} acc={acc:.4f}")
            correct_total, total = correct, seen

        model.eval()
        return correct_total / total if total else 0.0

    def _load_status(self) -> Dict[str, Any]:
        if STATUS_FILE.exists():
            try:
                with open(STATUS_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        return {
            "model_version":  "v_initial",
            "last_retrained": None,
            "accuracy":       None,
            "samples_used":   0,
        }

    def _save_status(self):
        with open(STATUS_FILE, "w") as f:
            json.dump(self._status, f, indent=2)
