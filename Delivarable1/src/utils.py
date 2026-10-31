import os
import joblib


def save_artifact(obj, file_path: str):
    """Save any picklable object (pipeline, model, background sample, ...) with joblib."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True) if os.path.dirname(file_path) else None
    joblib.dump(obj, file_path)


def load_artifact(file_path: str):
    """Load a joblib artifact."""
    return joblib.load(file_path)
