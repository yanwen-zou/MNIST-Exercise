import cupy as np

class Tensor:
    """
    Minimal autograd Tensor.
    Stores data, gradient, and backward function.
    """
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = None
        self._backward = lambda: None
        self._prev = set()

    def backward(self, grad=None):
        """
        Backpropagation through the computation graph.
        """
        if grad is None:
            grad = np.ones_like(self.data)

        self.grad = grad if self.grad is None else self.grad + grad

        topo = []
        visited = set()

        def build_topo(t):
            if t not in visited:
                visited.add(t)
                for child in t._prev:
                    build_topo(child)
                topo.append(t)

        build_topo(self)

        for t in reversed(topo):
            t._backward()


# ---------------------------
# Basic ops
# ---------------------------

def add(a: Tensor, b: Tensor):
    out = Tensor(a.data + b.data, requires_grad=a.requires_grad or b.requires_grad)

    def _backward():
        if a.requires_grad:
            a.grad = (a.grad if a.grad is not None else 0) + out.grad
        if b.requires_grad:
            # handle bias broadcasting
            grad_b = out.grad
            while grad_b.ndim > b.data.ndim:
                grad_b = grad_b.sum(axis=0)   # sum over extra dims
            for i in range(b.data.ndim):
                if b.data.shape[i] == 1 and grad_b.shape[i] > 1:
                    grad_b = grad_b.sum(axis=i, keepdims=True)
            b.grad = (b.grad if b.grad is not None else 0) + grad_b

    out._backward = _backward
    out._prev = {a, b}
    return out



def matmul(a: Tensor, b: Tensor):
    out = Tensor(a.data @ b.data, requires_grad=a.requires_grad or b.requires_grad)

    def _backward():
        if a.requires_grad:
            a.grad = (a.grad if a.grad is not None else 0) + out.grad @ b.data.T
        if b.requires_grad:
            b.grad = (b.grad if b.grad is not None else 0) + a.data.T @ out.grad

    out._backward = _backward
    out._prev = {a, b}
    return out


def relu(a: Tensor):
    out = Tensor(np.maximum(0, a.data), requires_grad=a.requires_grad)

    def _backward():
        if a.requires_grad:
            grad_mask = (a.data > 0).astype(np.float32)
            a.grad = (a.grad if a.grad is not None else 0) + grad_mask * out.grad

    out._backward = _backward
    out._prev = {a}
    return out


# ---------------------------
# Softmax + CrossEntropy
# ---------------------------

def softmax_crossentropy(logits: Tensor, labels: np.ndarray):
    exp = np.exp(logits.data - np.max(logits.data, axis=1, keepdims=True))
    probs = exp / np.sum(exp, axis=1, keepdims=True)

    N = logits.data.shape[0]
    loss_val = -np.log(probs[np.arange(N), labels] + 1e-9).mean()
    out = Tensor(loss_val, requires_grad=True)

    def _backward():
        grad_logits = probs.copy()
        grad_logits[np.arange(N), labels] -= 1
        grad_logits /= N
        if logits.requires_grad:
            logits.grad = (logits.grad if logits.grad is not None else 0) + grad_logits

    out._backward = _backward
    out._prev = {logits}
    return out


# ---------------------------
# Convolution (naive)
# ---------------------------

def conv2d(x: Tensor, W: Tensor, b: Tensor, stride=1, padding=0):
    """
    Naive Conv2D forward/backward using numpy.
    x: (batch, in_ch, H, W)
    W: (out_ch, in_ch, k, k)
    b: (out_ch,)
    """
    batch, in_ch, H, W_in = x.data.shape
    out_ch, _, k, _ = W.data.shape
    s = stride
    p = padding

    # compute output shape
    H_out = (H + 2*p - k) // s + 1
    W_out = (W_in + 2*p - k) // s + 1

    # pad input
    x_padded = np.pad(x.data, ((0,0),(0,0),(p,p),(p,p)), mode="constant")

    out_data = np.zeros((batch, out_ch, H_out, W_out), dtype=np.float32)
    for i in range(H_out):
        for j in range(W_out):
            region = x_padded[:, :, i*s:i*s+k, j*s:j*s+k]  # (batch, in_ch, k, k)
            out_data[:, :, i, j] = np.tensordot(region, W.data, axes=([1,2,3],[1,2,3])) + b.data

    out = Tensor(out_data, requires_grad=x.requires_grad or W.requires_grad or b.requires_grad)

    def _backward():
        if out.grad is None:
            return
        dx_padded = np.zeros_like(x_padded)
        dW = np.zeros_like(W.data)
        db = np.zeros_like(b.data)

        for i in range(H_out):
            for j in range(W_out):
                region = x_padded[:, :, i*s:i*s+k, j*s:j*s+k]
                for oc in range(out_ch):
                    dW[oc] += np.sum(region * out.grad[:, oc, i, j][:, None, None, None], axis=0)
                dx_padded[:, :, i*s:i*s+k, j*s:j*s+k] += np.tensordot(
                    out.grad[:, :, i, j], W.data, axes=(0,0)
                )
        db = np.sum(out.grad, axis=(0,2,3))

        if x.requires_grad:
            x.grad = (x.grad if x.grad is not None else 0) + dx_padded[:, :, p:p+H, p:p+W_in]
        if W.requires_grad:
            W.grad = (W.grad if W.grad is not None else 0) + dW
        if b.requires_grad:
            b.grad = (b.grad if b.grad is not None else 0) + db

    out._backward = _backward
    out._prev = {x, W, b}
    return out


# ---------------------------
# Max Pooling (naive)
# ---------------------------

def maxpool2d(x: Tensor, size=2, stride=2):
    """
    Naive MaxPool2D forward/backward.
    x: (batch, ch, H, W)
    """
    batch, ch, H, W_in = x.data.shape
    k, s = size, stride

    H_out = (H - k)//s + 1
    W_out = (W_in - k)//s + 1

    out_data = np.zeros((batch, ch, H_out, W_out), dtype=np.float32)
    mask = np.zeros_like(x.data, dtype=bool)

    for i in range(H_out):
        for j in range(W_out):
            region = x.data[:, :, i*s:i*s+k, j*s:j*s+k]
            max_val = np.max(region, axis=(2,3))
            out_data[:, :, i, j] = max_val
            mask_region = (region == max_val[:, :, None, None])
            mask[:, :, i*s:i*s+k, j*s:j*s+k] |= mask_region

    out = Tensor(out_data, requires_grad=x.requires_grad)

    def _backward():
        if x.requires_grad and out.grad is not None:
            dx = np.zeros_like(x.data)
            for i in range(H_out):
                for j in range(W_out):
                    dx[:, :, i*s:i*s+k, j*s:j*s+k] += out.grad[:, :, i, j][:, :, None, None] * \
                                                       mask[:, :, i*s:i*s+k, j*s:j*s+k]
            x.grad = (x.grad if x.grad is not None else 0) + dx

    out._backward = _backward
    out._prev = {x}
    return out
