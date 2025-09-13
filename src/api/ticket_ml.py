from fastapi import APIRouter
from pydantic import BaseModel
from ml.preprocess import prepare_data
from ml.model import TicketClassifier

router = APIRouter()

# Request body model
class TicketRequest(BaseModel):
    description: str

# Load or train the model once when API starts
clf = TicketClassifier()

# Example training with tiny dataset (replace with real training)
tickets = [
    "Cannot connect to VPN",
    "Laptop battery not charging",
    "Software installation failed",
    "Need access to email",
]
labels = ["network", "hardware", "software", "access"]
X_train, X_test, y_train, y_test, vectorizer = prepare_data(tickets, labels)
clf.train(X_train, y_train)

@router.post("/predict_ticket")
def predict_ticket(request: TicketRequest):
    vectorized = vectorizer.transform([request.description])
    pred = clf.model.predict(vectorized)  # <-- updated here
    return {"ticket_category": pred[0]}
