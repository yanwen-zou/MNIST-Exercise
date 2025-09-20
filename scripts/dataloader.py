import cupy as np
import os

class MNISTDataset:
    def __init__(self, csv_path, normalize=True):

        data = np.loadtxt(csv_path, delimiter=",", skiprows=1) 
        
        self.labels = data[:, 0].astype(int)
        self.images = data[:, 1:].astype(np.float32).reshape(-1, 28, 28)

        if normalize:
            self.images = self.images / 255.0

        self.num_samples = len(self.labels)

    def __len__(self):
        return self.num_samples

    def get_batch(self, batch_size=64, shuffle=True):
        """
        生成 mini-batch 数据
        """
        idxs = np.arange(self.num_samples)
        if shuffle:
            np.random.shuffle(idxs)

        for i in range(0, self.num_samples, batch_size):
            batch_idx = idxs[i:i+batch_size]
            yield self.images[batch_idx], self.labels[batch_idx]
