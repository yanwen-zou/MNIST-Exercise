from scripts.dataloader import MNISTDataset
from scripts.resnet9 import ResNet9
from scripts.train import train

if __name__ == "__main__":
    train_dataset = MNISTDataset("data/mnist_train.csv")
    test_dataset  = MNISTDataset("data/mnist_test.csv")

    model = ResNet9(num_classes=10)
    train(model, train_dataset, test_dataset, epochs=75, batch_size=256, lr=0.001)
