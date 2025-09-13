from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

def prepare_data(tickets, labels, test_size=0.2, random_state=42):
    """
    Preprocess ticket text data and split into train/test sets.
    
    Args:
        tickets (list[str]): List of ticket descriptions.
        labels (list[str]): Corresponding categories.
    
    Returns:
        X_train, X_test, y_train, y_test, vectorizer
    """
    vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
    X = vectorizer.fit_transform(tickets)
    X_train, X_test, y_train, y_test = train_test_split(
        X, labels, test_size=test_size, random_state=random_state
    )
    return X_train, X_test, y_train, y_test, vectorizer
