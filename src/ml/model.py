from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

MODEL_PATH = "src/ml/ticket_model.pkl"
VECTORIZER_PATH = "src/ml/vectorizer.pkl"

class TicketClassifier:
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000)
        self.vectorizer = None

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test):
        predictions = self.model.predict(X_test)
        acc = accuracy_score(y_test, predictions)
        report = classification_report(y_test, predictions)
        return acc, report

    def save(self, vectorizer):
        joblib.dump(self.model, MODEL_PATH)
        joblib.dump(vectorizer, VECTORIZER_PATH)

    def load(self):
        if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH):
            self.model = joblib.load(MODEL_PATH)
            self.vectorizer = joblib.load(VECTORIZER_PATH)
            return True
        return False
