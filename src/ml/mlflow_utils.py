import mlflow
import mlflow.sklearn

def start_run(run_name="TicketClassifier_Experiment"):
    """
    Start a new MLflow run.
    """
    mlflow.start_run(run_name=run_name)

def log_model(model, artifact_path="model"):
    """
    Log a scikit-learn model to MLflow.
    """
    mlflow.sklearn.log_model(model, artifact_path=artifact_path)

def log_params(params: dict):
    """
    Log hyperparameters.
    """
    for k, v in params.items():
        mlflow.log_param(k, v)

def log_metrics(metrics: dict):
    """
    Log metrics like accuracy, F1-score, etc.
    """
    for k, v in metrics.items():
        mlflow.log_metric(k, v)

def end_run():
    mlflow.end_run()
