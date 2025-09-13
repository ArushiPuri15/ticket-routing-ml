import sys
import os

# Add src/ to Python path so imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

# Imports after path adjustment
from ml.preprocess import prepare_data
from ml.model import TicketClassifier

# Dummy test to check pytest discovery
def test_dummy():
    assert True

# Actual ML test
def test_ticket_classifier_training():
    tickets = [
        "Cannot connect to VPN",
        "Laptop battery not charging",
        "Software installation failed",
        "Need access to email",
    ]
    labels = ["network", "hardware", "software", "access"]

    # Preprocess data
    X_train, X_test, y_train, y_test, vectorizer = prepare_data(tickets, labels)

    # Train classifier
    clf = TicketClassifier()
    clf.train(X_train, y_train)

    # Evaluate
    acc, report = clf.evaluate(X_test, y_test)

    assert acc >= 0.0  # just check model runs
