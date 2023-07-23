import jax
import jax.numpy as jnp

from plotly.subplots import make_subplots
import plotly.graph_objects as go

import optax

from tqdm.notebook import tqdm
from functools import partial

import json
import pickle
from time import time

import numpy as np
from sklearn.decomposition import IncrementalPCA

class LandscapeProblem():
  def __init__(self, model, batch_size=64):
    self.batch_size = batch_size
    self.load_dataset()
    self.model = model

  def dataset(self, idx):
    raise NotImplementedError()

  def dataset_len(self):
    return -1

  def load_dataset(self):
    raise NotImplementedError()

  @partial(jax.jit, static_argnums=(0,))
  def eval_params(self, params, batch, label):
    raise NotImplementedError()

  def train_path(self, optimizer, starting_params=None, epochs=50, log_name='model', single_sample=False,  save=False, **kwargs):
    params_path = []
    params = self.model.init(jax.random.PRNGKey(0), self.dataset(0)[0]) if starting_params == None else starting_params
    opt_state = optimizer.init(params)
    loss_value = -1

    @jax.jit
    def step(params, opt_state, batch, label):
      loss_value, grads = jax.value_and_grad(self.eval_params)(params, batch, label)
      updates, opt_state = optimizer.update(grads, opt_state, params)
      params = optax.apply_updates(params, updates)
      return params, opt_state, loss_value

    total_epochs = epochs if not single_sample else 3
    
    for epoch in range(total_epochs):
      print('Epoch %d/%d' % (epoch + 1, total_epochs))
      dataset_iter = list(range(self.dataset_len()))
      np.random.shuffle(dataset_iter)
      dataset_iter = tqdm(dataset_iter)
      for i in dataset_iter:
        batch, label = self.dataset(i)
        params, opt_state, loss_value = step(params, opt_state, batch, label)
        dataset_iter.set_description('Loss: %f' % loss_value)
        if single_sample:
          params_path.append(params)
      
      if not single_sample:
        params_path.append(params)

      if save:
        with open('/content/drive/MyDrive/SFI/%s-%d.pkl' % (log_name, int(time())), 'wb') as param_pkl:
          pickle.dump(params_path, param_pkl)

    return params_path

class LossVisualizer():
  def flatten_path(self, parameter_path):
    _, unravel = jax.flatten_util.ravel_pytree(parameter_path[0])
    self.unravel = unravel
    param_vec = [jax.flatten_util.ravel_pytree(params)[0] for params in parameter_path]
    param_vec = jnp.stack(param_vec)
    return param_vec

  @partial(jax.jit, static_argnums=(0,1))
  def plot_path(self,
                landscape,
                param_vec,
                batch,
                label):
    return jax.vmap(lambda params: landscape.eval_params(self.unravel(params), batch, label), in_axes=0, out_axes=0)(param_vec)

  @partial(jax.jit, static_argnums=(0,1))
  def plot_surface(self, 
                   landscape, 
                   base_vec, 
                   x_vec,
                   y_vec,
                   row_deltas,
                   lowdim_path,
                   batch,
                   label):
    return jax.vmap(
      lambda delta: landscape.eval_params(self.unravel(base_vec + delta[0]*x_vec + delta[1]*y_vec), batch, label), 
      in_axes=0, out_axes=0
    )(row_deltas)

  def project(self, param_vec):
    pca = IncrementalPCA(n_components=2)
    pca.fit(param_vec)
    self.lowdim_fit = pca
    # print(pca.explained_variance_ratio_)
    return pca
  
  def mill_plot(self, landscape, parameter_path, filename, **kwargs):
    surface_data = self.process(landscape, parameter_path, **kwargs)
    output = {
      'path': {
        'bounds': [[float(jnp.amin(surface_data['loss_x'])), float(jnp.amax(surface_data['loss_x']))],
                   [float(jnp.amin(surface_data['loss_y'])), float(jnp.amax(surface_data['loss_y']))],
                   [float(jnp.amin(surface_data['loss_z'])), float(jnp.amax(surface_data['loss_z']))]],
        'data': jnp.stack([surface_data['loss_x'], surface_data['loss_y'], surface_data['loss_z']], axis=1).tolist(),
        'directions': [[0,0,1],[1,0,0]],
        'notes': ['Best Loss: %d' % jnp.amin(surface_data['loss_z'])]
      },
      'surface': {
        'bounds': [[float(jnp.amin(surface_data['x'])), float(jnp.amax(surface_data['x']))],
                   [float(jnp.amin(surface_data['y'])), float(jnp.amax(surface_data['y']))],
                   [float(jnp.amin(surface_data['z'])), float(jnp.amax(surface_data['z']))]],
        'data': surface_data['z'].tolist(),
        'directions': [[0,0,1],[1,0,0]],
        'notes': ['No notes yet.']
      }
    }

    with open(filename, 'w') as output_fp:
      json.dump(output, output_fp)
      print('Wrote JSON output to %s.' % filename)

  def plotly_plot(self, landscape, parameter_path, **kwargs):
    surface_data = self.process(landscape, parameter_path, **kwargs)

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{'type': 'surface'}, {'type': 'xy'}]]
      )
    
    fig.add_trace(
      go.Surface(
        x=surface_data['x'], 
        y=surface_data['y'], 
        z=surface_data['z'],
        colorscale='viridis'
      ),
      1, 1
    )
    
    fig.add_trace(
      go.Scatter3d(
          x=surface_data['loss_x'],
          y=surface_data['loss_y'],
          z=surface_data['loss_z'],
          marker=dict(size=4, color='red'),
          line=dict(color='orange', width=2)
      ),
      1, 1
    )

    fig.add_trace(
      go.Scatter(
          x=surface_data['loss_x'],
          y=surface_data['loss_y'],
          marker=dict(size=4, color='red'),
          line=dict(color='red', width=2)
      ),
      1, 2
    )
    fig.add_trace(
        go.Contour(
          z=surface_data['z'],
          x0=float(surface_data['x'][0,0]),
          dx=float(surface_data['x'][0,1] - surface_data['x'][0,0]),
          y0=float(surface_data['y'][0,0]),
          dy=float(surface_data['y'][1,0] - surface_data['y'][0,0]),
          line_smoothing=0.85,
          colorscale='viridis'
        ), 1, 2
    )

    fig.update_layout(width=1500, height=800)

    return fig

  def filter_norm(self, component, params_dict):
    component_dict = self.unravel(component)
    component_dict = jax.tree_util.tree_map(lambda c: c / jnp.linalg.norm(c), component_dict)
    component_dict = jax.tree_util.tree_map(lambda c, p: c * jnp.linalg.norm(p), component_dict, params_dict)
    return jax.flatten_util.ravel_pytree(component_dict)[0]

  def process(self, 
              landscape, 
              parameter_path, 
              x_range=None, 
              y_range=None,
              x_vec=None,
              y_vec=None,
              resolution=101, 
              new_fit=True, 
              margin_factor=1.5,
              filter_norm=False,
              verbose=True, 
              sample_idx=0):
    param_vec = self.flatten_path(parameter_path)

    if new_fit: self.base_vec = param_vec[-1,:][None,...]
    param_delta = param_vec - self.base_vec

    batch, label = landscape.dataset(sample_idx)

    if new_fit: self.project(param_delta[1:-1,:])

    if filter_norm:
      for i in range(2):
        self.lowdim_fit.components_[i] = self.filter_norm(self.lowdim_fit.components_[i], parameter_path[-1])

    lowdim_path = self.lowdim_fit.transform(param_delta)

    x_amp = margin_factor * np.amax(np.abs(lowdim_path[:,0])) if x_range == None else x_range
    y_amp = margin_factor * np.amax(np.abs(lowdim_path[:,1])) if y_range == None else y_range
    # x_amp = y_amp = margin_factor
    x_mesh, y_mesh = jnp.meshgrid(jnp.linspace(-x_amp, x_amp, resolution), jnp.linspace(-y_amp, y_amp, resolution))

    deltas = jnp.stack([x_mesh, y_mesh], axis=2)
    if verbose: print(deltas.shape)

    param_losses = self.plot_path(landscape, param_vec, batch, label)

    dir_x = self.lowdim_fit.components_[0] if x_vec is None else x_vec
    dir_y = self.lowdim_fit.components_[1] if y_vec is None else y_vec

    Z = jax.lax.map(lambda row_deltas:
      self.plot_surface(
          landscape,
          param_vec[-1,:], 
          dir_x,
          dir_y,
          row_deltas,
          lowdim_path,
          batch,
          label
      ), 
    deltas)
    # Z = jnp.sin(5 * x_mesh) + jnp.sin(5 * y_mesh)

    if verbose: print(Z.shape)

    return {
        'x': x_mesh,
        'y': y_mesh,
        'z': Z,
        'loss_x': lowdim_path[:,0],
        'loss_y': lowdim_path[:,1],
        'loss_z': param_losses,
        'dir_x': dir_x,
        'dir_y': dir_y,
        'base_vec': param_vec[-1,:],
        'lowdim_path': lowdim_path
    }
