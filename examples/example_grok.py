from losscape.plot import *
from losscape.models import *

import torch
import torchvision
from torch.utils.data import DataLoader
import math

mnist_data = torchvision.datasets.MNIST('./sample_data/', download=True)
pil_transform = torchvision.transforms.PILToTensor()

class GrokMNIST(LandscapeProblem):
    def dataset(self, idx):
        batch_input, batch_label = [], []
        for i in range(32):
            sample, label = mnist_data[10*idx + i]
            label_vector = torch.zeros(10)
            label_vector[label] = 1.
            batch_input.append(pil_transform(sample).flatten().numpy())
            batch_label.append(label_vector.numpy())
        return jnp.stack(batch_input), jnp.stack(batch_label)

    def dataset_len(self):
        return 1000

    def load_dataset(self):
        pass

    def eval_params(self, params, batch, label):
        prediction = self.model.apply(params, batch)
        return optax.l2_loss(jnp.squeeze(prediction), label).mean()
    
class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
      x = nn.Dense(784)(x)
      x = nn.relu(x)
      x = nn.Dense(200)(x)
      x = nn.relu(x)
      x = nn.Dense(200)(x)
      x = nn.relu(x)
      x = nn.Dense(200)(x)
      x = nn.relu(x)
      x = nn.Dense(10)(x)
      return nn.softmax(x)

landscape = GrokMNIST(model=MLP())
optimizer = optax.adamw(1e-4)
parameter_path = landscape.train_path(optimizer, epochs=30, test_idx=3000)

vis = LossVisualizer()
surface_data = vis.process(landscape, parameter_path[:], resolution=101, margin_factor=10, y_range=8)

vis.mill_plot(landscape, parameter_path, filename='data_mnist_train.json', resolution=101, margin_factor=10, sample_idx=0)
vis.mill_plot(landscape, parameter_path, filename='data_mnist_test.json', resolution=101, margin_factor=10, sample_idx=3000)

# fig = vis.plotly_plot(landscape, parameter_path, resolution=101, margin_factor=100, sample_idx=3000)
# fig.write_html('data_mnist_test.html')

# fig = vis.plotly_plot(landscape, parameter_path, resolution=101, margin_factor=100, sample_idx=0)
# fig.write_html('data_mnist_train.html')
