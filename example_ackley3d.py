from plot import *
from models import *

import matplotlib.pyplot as plt

def ackley(x, y, z):
  a = 2
  b = 0.2
  c = 2*jnp.pi
  d = 3
  return -a * jnp.exp(
    -b * jnp.sqrt((1/d) * (jnp.square(x) + jnp.square(y) + jnp.square(z)))
  ) - jnp.exp(
    (1/d) * (jnp.cos(c * x) + jnp.cos(c * y) + jnp.cos(c * z))
  ) + jnp.e + a

class AckleyModel(nn.Module):
  @nn.compact
  def __call__(self, inputs):
    x = self.param('x', nn.initializers.constant(-3), (1,))
    y = self.param('y', nn.initializers.constant(3), (1,))
    z = self.param('z', nn.initializers.constant(5), (1,))
    return ackley(x, y, z)

class AckleyLandscape(LandscapeProblem):
  def dataset(self, idx):
    return None, None

  def dataset_len(self):
    return 100

  def load_dataset(self):
    pass

  def eval_params(self, params, batch, label):
    return self.model.apply(params, None).mean()
  
model = AckleyModel()
landscape = AckleyLandscape(model)
optimizer = optax.adagrad(1e-3)
parameter_path = landscape.train_path(optimizer, epochs=160,)

vis = LossVisualizer()
surface_data = vis.process(landscape, parameter_path[:], resolution=101, margin_factor=10, y_range=8)
vis.mill_plot(landscape, parameter_path, filename='data_ackley3d.json', resolution=101, margin_factor=10)

# gap_dist = vis.uniform_gap_distribution(landscape, parameter_path, samples=1500)
# # print(gap_dist)
# plt.hist(gap_dist, bins=30)
# plt.savefig('hist_ackley3d.png')

# fig = vis.plotly_plot(landscape, parameter_path[:], resolution=101, margin_factor=2e4)
# print(parameter_path[-1])
# print(vis.lowdim_fit.components_)
# fig.write_html('render.html')