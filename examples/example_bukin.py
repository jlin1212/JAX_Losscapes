from losscape.plot import *
from losscape.models import *

def bukin(x, y):
  return 100 * jnp.sqrt(jnp.abs(y - 0.01 * (x ** 2)) + 0.01 * jnp.abs(x + 10))

class BukinModel(nn.Module):
  @nn.compact
  def __call__(self, inputs):
    x = self.param('x', nn.initializers.constant(8), (1,))
    y = self.param('y', nn.initializers.constant(3), (1,))
    return bukin(x, y)

class BukinLandscape(LandscapeProblem):
  def dataset(self, idx):
    return None, None

  def dataset_len(self):
    return 100

  def load_dataset(self):
    pass

  def eval_params(self, params, batch, label):
    return self.model.apply(params, None).mean()
  
model = BukinModel()
landscape = BukinLandscape(model)
optimizer = optax.adam(1e-3)
parameter_path = landscape.train_path(optimizer, epochs=60,)

vis = LossVisualizer()
surface_data = vis.process(landscape, parameter_path[:], resolution=101, margin_factor=2e4, y_range=8)

# print(parameter_path[-1])
# print(vis.lowdim_fit.components_)

# vis.mill_plot(landscape, parameter_path, filename='data_bukin.json', resolution=101, margin_factor=1, y_range=8)

fig = vis.plotly_plot(landscape, parameter_path[:], resolution=101, margin_factor=1, y_range=8)
fig.write_html('render.html')