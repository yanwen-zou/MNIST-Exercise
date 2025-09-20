import time
import matplotlib.pyplot as plt
import os
import numpy as np
from scripts.dataloader import MNISTDataset
from scripts.resnet9 import ResNet9
from scripts.autograd import Tensor
from scripts.utils import Adam

def profile_training_step():
    # load a small batch
    dataset = MNISTDataset("data/mnist_train.csv")
    X_batch, y_batch = next(dataset.get_batch(batch_size=64))
    X = Tensor(X_batch[:, None, :, :], requires_grad=False)
    y = y_batch

    model = ResNet9(num_classes=10)
    optimizer = Adam(model.parameters(), lr=0.001)

    # measure forward
    t0 = time.perf_counter()
    loss = model.forward(X, y)
    t1 = time.perf_counter()

    # measure backward
    loss.backward()
    t2 = time.perf_counter()

    # measure optimizer
    optimizer.step()
    t3 = time.perf_counter()

    times = {
        "Forward": (t1 - t0),
        "Backward": (t2 - t1),
        "Optimizer": (t3 - t2)
    }

    print("⏱️ Performance profile:")
    for k, v in times.items():
        print(f"{k}: {v*1000:.2f} ms")

    # plot bar chart
    os.makedirs("experiments/curves", exist_ok=True)
    plt.figure()
    plt.bar(times.keys(), [v*1000 for v in times.values()])
    plt.ylabel("Time (ms)")
    plt.title("Training Step Performance Profile")
    plt.savefig("experiments/curves/profile.png")
    print("✅ Saved profile chart to experiments/curves/profile.png")

if __name__ == "__main__":
    profile_training_step()
