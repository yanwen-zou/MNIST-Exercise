import cupy as np
from scripts.autograd import Tensor, matmul, add, relu, conv2d, maxpool2d

# ---------------------------
# Fully Connected Layer
# ---------------------------
class Linear:
    def __init__(self, in_features, out_features):
        limit = np.sqrt(6 / (in_features + out_features))
        self.W = Tensor(
            np.random.uniform(-limit, limit, (in_features, out_features)),
            requires_grad=True
        )
        self.b = Tensor(np.zeros(out_features), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        return add(matmul(x, self.W), self.b)


# ---------------------------
# ReLU Activation
# ---------------------------
class ReLU:
    def forward(self, x: Tensor) -> Tensor:
        return relu(x)


# ---------------------------
# Softmax + CrossEntropy Loss
# ---------------------------
class SoftmaxCrossEntropy:
    def forward(self, logits: Tensor, labels: np.ndarray) -> Tensor:
        from scripts.autograd import softmax_crossentropy
        return softmax_crossentropy(logits, labels)


# ---------------------------
# Convolution Layer
# ---------------------------
class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        self.stride = stride
        self.padding = padding
        self.kernel_size = kernel_size

        limit = np.sqrt(6 / (in_channels * kernel_size * kernel_size + out_channels))
        self.W = Tensor(
            np.random.uniform(-limit, limit, 
                              (out_channels, in_channels, kernel_size, kernel_size)),
            requires_grad=True
        )
        self.b = Tensor(np.zeros(out_channels), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        return conv2d(x, self.W, self.b, stride=self.stride, padding=self.padding)


# ---------------------------
# Max Pooling Layer
# ---------------------------
class MaxPool2D:
    def __init__(self, size=2, stride=2):
        self.size = size
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        return maxpool2d(x, size=self.size, stride=self.stride)
