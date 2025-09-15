# src/api/ticket_ml.py
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from src.ml.serve import TicketClassifier
import logging

LOG = logging.getLogger(__name__)

router = APIRouter()

class TicketRequest(BaseModel):
    subject: Optional[str] = Field("", description="Ticket subject (optional)")
    body: str = Field(..., description="Ticket body / description")

class SinglePrediction(BaseModel):
    label: str
    label_id: Optional[int] = None
    confidence: Optional[float] = None

class TicketResponse(BaseModel):
    predictions: List[SinglePrediction]

@router.post("/predict_ticket", response_model=TicketResponse)
def predict_ticket(req: TicketRequest):
    try:
        # combine subject + body following preprocessing design
        text = f"{(req.subject or '').strip()} {req.body.strip()}".strip()
        if not text:
            raise HTTPException(status_code=400, detail="Empty text provided")

        # use classifier
        preds = TicketClassifier.predict_texts([text])
        # preds is list of dicts
        return TicketResponse(predictions=[SinglePrediction(**p) for p in preds])
    except Exception as e:
        LOG.exception("Error during prediction: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
