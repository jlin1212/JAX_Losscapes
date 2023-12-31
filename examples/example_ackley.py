from losscape.plot import *
from losscape.models import *

def ackley(x, y):
  a = 20
  b = 0.2
  c = 2*jnp.pi
  d = 2
  return -a * jnp.exp(
    -b * jnp.sqrt((1/d) * (jnp.square(x) + jnp.square(y)))
  ) - jnp.exp(
    (1/d) * (jnp.cos(c * x) + jnp.cos(c * y))
  ) + jnp.e + a

class AckleyModel(nn.Module):
  @nn.compact
  def __call__(self, inputs):
    x = self.param('x', nn.initializers.constant(-3), (1,))
    y = self.param('y', nn.initializers.constant(3), (1,))
    return ackley(x, y)

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
optimizer = optax.adam(1)
parameter_path = landscape.train_path(optimizer, epochs=60,)

vis = LossVisualizer()
surface_data = vis.process(landscape, parameter_path[:], resolution=101, margin_factor=2e4, y_range=8)
vis.mill_plot(landscape, parameter_path, filename='data_ackley.json', resolution=101, margin_factor=2e4, y_range=8)

# fig = vis.plotly_plot(landscape, parameter_path[:], resolution=101, margin_factor=2e4, y_range=8)
# print(parameter_path[-1])
# print(vis.lowdim_fit.components_)
# fig.write_html('render.html')