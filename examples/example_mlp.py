from losscape.plot import *
from losscape.models import *

import math

# Define the model setup - how the dataset is loaded, its length, and the way each actual sample is
# returned.

class CaliforniaHousingProblem(LandscapeProblem):
  def dataset(self, idx):
    data = self.data[idx*self.batch_size:(idx+1)*self.batch_size,:]
    return data[:,:8], data[:,8]

  def dataset_len(self):
    return math.floor(self.data.shape[0] / self.batch_size)

  def load_dataset(self):
    california_data = np.loadtxt('./sample_data/california_housing_train.csv', skiprows=1, delimiter=',')
    np.random.shuffle(california_data)
    california_data = california_data - np.mean(california_data, axis=0, keepdims=True)
    california_min = np.amin(california_data, axis=0, keepdims=True)
    california_max = np.amax(california_data, axis=0, keepdims=True)
    california_data = (california_data - california_min) / (california_max - california_min)
    self.data = jnp.array(california_data)

  def eval_params(self, params, batch, label):
    prediction = self.model.apply(params, batch[:,:8])
    return optax.l2_loss(jnp.squeeze(prediction), label).mean()

# Initialize problem statement / optimizer, etc.
# Then, extract the "path" by training the model.

model = MicroMLP()
chmlp = CaliforniaHousingProblem(model=model, batch_size=64)
optimizer = optax.MultiSteps(optax.sgd(learning_rate=2e-3), 2)
parameter_path = chmlp.train_path(optimizer, epochs=6, single_sample=False)

# Visualize the resulting landscape

loss_vis = LossVisualizer()
surface_data = loss_vis.process(chmlp, parameter_path, resolution=101)
fig = loss_vis.plotly_plot(chmlp, parameter_path, resolution=101, sample_idx=10, margin_factor=1, y_range=0.1, filter_norm=False)
fig.write_html('render.html')
