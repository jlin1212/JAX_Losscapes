from losscape.plot import *
from losscape.models import *

import math

import torchvision
import matplotlib.pyplot as plt

cifar_data = torchvision.datasets.CIFAR10('sample_data/cifar/', download=True)

plt.imshow(cifar_data[1000][0])
plt.show()

CIFAR_MEAN = [0.4913997551666284, 0.48215855929893703, 0.4465309133731618]
CIFAR_STD = [0.24703225141799082, 0.24348516474564, 0.26158783926049628]

class CifarResnet(LandscapeProblem):
  def dataset(self, idx):
    img, label = self.data[idx]
    img = jnp.array(img) / 255.
    img = (img - jnp.array(CIFAR_MEAN)[None,None,...]) / jnp.array(CIFAR_STD)[None,None,...]
    img = img[None,...]
    return img, label

  def dataset_len(self):
    return math.floor(len(cifar_data) / 10)

  def load_dataset(self):
    self.data = cifar_data

  def eval_params(self, params, batch, label):
    logits = self.model.apply(params, batch)
    return optax.softmax_cross_entropy_with_integer_labels(logits, label[None,...]).mean()

cifar_landscape = CifarResnet(ResNet50(num_classes=10), batch_size=2)
optimizer = optax.MultiSteps(optax.adam(1e-4), 2)
parameter_path = cifar_landscape.train_path(optimizer, epochs=10)

fig = LossVisualizer().plotly_plot(cifar_landscape, parameter_path, resolution=101)
fig.show()
