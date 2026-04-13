# routes/pipeline_routes.py
#
# Adds the /api/pipeline/retrain endpoint that the Node.js service calls
# when a vet-approved batch is ready.  Also exposes a status endpoint.

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import asyncio
import logging

from services.retraining_service import RetrainingService

logger   = logging.getLogger(__name__)
router   = APIRouter(prefix="/pipeline", tags=["Retraining Pipeline"])
_service = RetrainingService()


# ── Request / Response schemas ────────────────────────────────────────────────

class SampleMetadata(BaseModel):
    cowId:     Optional[str] = None
    createdAt: Optional[str] = None
    inputMeta: Optional[Dict[str, Any]] = {}

class TrainingSample(BaseModel):
    id:                  str
    label:               str          # "FMD" | "LSD" | "Healthy"
    confidence:          float
    crossModalAgreement: float
    imageConfidence:     Optional[float] = None
    symptomPrediction:   Optional[str]   = None
    bloodPrediction:     Optional[str]   = None
    metadata:            Optional[SampleMetadata] = None

class ThresholdsUsed(BaseModel):
    confidence: float
    crossModal: float

class RetrainRequest(BaseModel):
    samples:         List[TrainingSample]
    totalSamples:    int
    triggeredAt:     str
    thresholdsUsed:  ThresholdsUsed

class RetrainResponse(BaseModel):
    success:           bool
    model_version:     str
    accuracy:          Optional[float] = None
    previous_accuracy: Optional[float] = None
    message:           str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/retrain", response_model=RetrainResponse)
async def trigger_retraining(payload: RetrainRequest):
    """
    Called by the Node.js HybridRetrainingService after vet approval.
    Runs the actual model fine-tuning in a background thread so the HTTP
    response returns quickly.
    """
    if not payload.samples:
        raise HTTPException(status_code=400, detail="No samples provided for retraining")

    logger.info(f"[Pipeline] Retraining triggered with {payload.totalSamples} samples")

    try:
        result = await asyncio.get_event_loop().run_in_executor(
            None,
            _service.retrain,
            payload.dict()
        )
        return RetrainResponse(
            success=True,
            model_version=result["model_version"],
            accuracy=result.get("accuracy"),
            previous_accuracy=result.get("previous_accuracy"),
            message=f"Retraining complete. New model: {result['model_version']}"
        )
    except Exception as e:
        logger.error(f"[Pipeline] Retraining failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
async def pipeline_status():
    """Returns the current model version and last retraining info."""
    return _service.get_status()


@router.get("/batches")
async def pipeline_batches():
    """Compatibility endpoint for UI clients expecting a batches list."""
    return {
        "batches": [],
        "status": _service.get_status(),
    }
