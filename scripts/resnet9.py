import cupy as np
from scripts.layers import Conv2D, ReLU, MaxPool2D, Linear, SoftmaxCrossEntropy
from scripts.autograd import Tensor, add

# ---------------------------
# Residual Block
# ---------------------------
class ResidualBlock:
    def __init__(self, channels):
        """
        Residual Block: Conv -> ReLU -> Conv + skip -> ReLU
        in/out channels are the same
        """
        self.conv1 = Conv2D(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu1 = ReLU()
        self.conv2 = Conv2D(channels, channels, kernel_size=3, stride=1, padding=1)
        self.relu2 = ReLU()

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv1.forward(x)
        out = self.relu1.forward(out)
        out = self.conv2.forward(out)
        out = add(out, x)
        out = self.relu2.forward(out)
        return out

    def params(self):
        return [self.conv1.W, self.conv1.b,
                self.conv2.W, self.conv2.b]


# ---------------------------
# ResNet9 Model
# ---------------------------
class ResNet9:
    def __init__(self, num_classes=10):
        """
        ResNet9 adapted for MNIST (1 x 28 x 28 input)
        """
        # Stage 1
        self.conv1 = Conv2D(1, 64, kernel_size=3, stride=1, padding=1)
        self.relu1 = ReLU()
        self.conv2 = Conv2D(64, 128, kernel_size=3, stride=1, padding=1)
        self.relu2 = ReLU()
        self.pool1 = MaxPool2D(2, 2)

        # Residual Block 1
        self.res1 = ResidualBlock(128)

        # Stage 2
        self.conv3 = Conv2D(128, 256, kernel_size=3, stride=1, padding=1)
        self.relu3 = ReLU()
        self.pool2 = MaxPool2D(2, 2)

        self.conv4 = Conv2D(256, 512, kernel_size=3, stride=1, padding=1)
        self.relu4 = ReLU()
        self.pool3 = MaxPool2D(2, 2)

        # Residual Block 2
        self.res2 = ResidualBlock(512)

        # Fully Connected
        self.fc = Linear(512, num_classes)

        # Loss
        self.criterion = SoftmaxCrossEntropy()

    def forward(self, x: Tensor, labels=None) -> Tensor:
        """
        Forward pass
        x: Tensor (batch, 1, 28, 28)
        labels: np.ndarray (batch,)
        """
        # Stage 1
        out = self.conv1.forward(x)
        out = self.relu1.forward(out)
        out = self.conv2.forward(out)
        out = self.relu2.forward(out)
        out = self.pool1.forward(out)

        # Residual Block 1
        out = self.res1.forward(out)

        # Stage 2
        out = self.conv3.forward(out)
        out = self.relu3.forward(out)
        out = self.pool2.forward(out)

        out = self.conv4.forward(out)
        out = self.relu4.forward(out)
        out = self.pool3.forward(out)

        # Residual Block 2
        out = self.res2.forward(out)

        # Global Average Pooling (batch, 512, H, W) -> (batch, 512)
        out = Tensor(out.data.mean(axis=(2, 3)), requires_grad=True)

        # Fully Connected
        logits = self.fc.forward(out)

        if labels is not None:
            loss = self.criterion.forward(logits, labels)
            return loss
        else:
            return logits

    def parameters(self):
        """
        Return all trainable parameters
        """
        params = []
        for layer in [self.conv1, self.conv2, self.conv3, self.conv4, self.fc]:
            params.extend([layer.W, layer.b])
        params.extend(self.res1.params())
        params.extend(self.res2.params())
        return params
