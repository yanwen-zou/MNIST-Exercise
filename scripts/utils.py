import cupy as np

class Adam:
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), eps=1e-8):
        self.params = params
        self.lr = lr
        self.betas = betas
        self.eps = eps
        self.t = 0  # time step

        # Initialize moment estimates
        self.m = [np.zeros_like(p.data) for p in params]
        self.v = [np.zeros_like(p.data) for p in params]

    def step(self):
        self.t += 1
        beta1, beta2 = self.betas

        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            g = p.grad

            # Update biased first moment estimate
            self.m[i] = beta1 * self.m[i] + (1 - beta1) * g
            # Update biased second raw moment estimate
            self.v[i] = beta2 * self.v[i] + (1 - beta2) * (g * g)

            # Bias correction
            m_hat = self.m[i] / (1 - beta1 ** self.t)
            v_hat = self.v[i] / (1 - beta2 ** self.t)

            # Update parameters
            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

            # Reset gradient
            p.grad = None
