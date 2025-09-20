import pickle
import os

def save_model(model, path="experiments/resnet9.pkl"):
    """
    Save model parameters to a file.
    model: instance of ResNet9 (or any class with .parameters())
    path: save file path
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    params = [p.data for p in model.parameters()]
    with open(path, "wb") as f:
        pickle.dump(params, f)
    print(f"✅ Model saved to {path}")


def load_model(model, path="experiments/resnet9.pkl"):
    """
    Load model parameters from file and copy into model.
    model: instance of ResNet9
    path: checkpoint file
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"No checkpoint found at {path}")

    with open(path, "rb") as f:
        saved_params = pickle.load(f)

    for p, w in zip(model.parameters(), saved_params):
        p.data[...] = w  # copy numpy array back into Tensor

    print(f"✅ Model loaded from {path}")
    return model
