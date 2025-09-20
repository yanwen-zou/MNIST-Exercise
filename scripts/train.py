from scripts.dataloader import MNISTDataset
from scripts.resnet9 import ResNet9
from scripts.autograd import Tensor
from scripts.utils import Adam
from scripts.checkpoint import save_model
import cupy as np
import time
import matplotlib.pyplot as plt
import os

train_dataset = MNISTDataset(csv_path="./data/mnist_train.csv")
test_dataset = MNISTDataset(csv_path="./data/mnist_test.csv")

model = ResNet9(num_classes=10)

def train(model, train_dataset, test_dataset, epochs=5, batch_size=64, lr=0.001):
    optimizer = Adam(model.parameters(), lr=lr)

    train_losses = []   # record average loss per epoch
    test_accuracies = []  # optional, could also plot acc curve
    epoch_times = []

    for epoch in range(epochs):
        epoch_start = time.perf_counter()

        # ---------------------------
        # Training
        # ---------------------------
        losses = []
        correct = 0
        total = 0
        for X_batch, y_batch in train_dataset.get_batch(batch_size=batch_size):
            # Convert batch to Tensor (add channel dim: 1x28x28)
            X = Tensor(X_batch[:, None, :, :], requires_grad=False)
            y = y_batch

            # Forward
            loss = model.forward(X, y)
            losses.append(float(loss.data))  # ensure float

            # Backward
            loss.backward()

            # Update
            optimizer.step()

            # Accuracy (training set)
            logits = model.forward(X).data
            preds = np.argmax(logits, axis=1)
            correct += int((preds == y).sum())
            total += len(y)

        train_acc = correct / total
        avg_loss = sum(losses) / len(losses)
        train_losses.append(avg_loss)

        # ---------------------------
        # Evaluation
        # ---------------------------
        correct, total = 0, 0
        for X_batch, y_batch in test_dataset.get_batch(batch_size=batch_size, shuffle=False):
            X = Tensor(X_batch[:, None, :, :], requires_grad=False)
            logits = model.forward(X).data
            preds = np.argmax(logits, axis=1)
            correct += int((preds == y_batch).sum())
            total += len(y_batch)
        test_acc = correct / total
        test_accuracies.append(test_acc)

        epoch_end = time.perf_counter()
        epoch_time = epoch_end - epoch_start
        epoch_times.append(epoch_time)

        print(f"Epoch {epoch+1}/{epochs} "
              f"- Loss: {avg_loss:.4f} "
              f"- Train Acc: {train_acc:.4f} "
              f"- Test Acc: {test_acc:.4f} "
              f"- Time: {epoch_time:.2f}s")

    # ---------------------------
    # Save curves
    # ---------------------------
    os.makedirs("experiments/curves", exist_ok=True)

    # Loss curve
    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, label="Train Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.savefig("experiments/curves/loss_curve.png")
    print("✅ Saved loss curve to experiments/curves/loss_curve.png")

    # (optional) Accuracy curve
    plt.figure()
    plt.plot(range(1, epochs+1), test_accuracies, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Test Accuracy Curve")
    plt.legend()
    plt.savefig("experiments/curves/acc_curve.png")
    print("✅ Saved accuracy curve to experiments/curves/acc_curve.png")

    # (optional) Epoch time curve
    plt.figure()
    plt.plot(range(1, epochs+1), epoch_times, label="Epoch Time (s)")
    plt.xlabel("Epoch")
    plt.ylabel("Time (s)")
    plt.title("Epoch Time Curve")
    plt.legend()
    plt.savefig("experiments/curves/epoch_time.png")
    print("✅ Saved epoch time curve to experiments/curves/epoch_time.png")

    # ---------------------------
    # Save model checkpoint
    # ---------------------------
    save_model(model, "experiments/resnet9.pkl")
