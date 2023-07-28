
import jax
import jax.numpy as jnp
import flax

from plotly.subplots import make_subplots
import plotly.graph_objects as go

import optax

from tqdm import tqdm
from functools import partial

import json
import pickle
from time import time

import numpy as np
from sklearn.decomposition import IncrementalPCA

import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

def pathLength(path):
  sumDist = 0
  for i in range(path.shape[0]-1):
    sumDist += np.linalg.norm(path[i]-path[i+1])  
  return sumDist

def pathLengthProg(path):
  sumDist = 0
  dists = [0]
  for i in range(path.shape[0]-1):    
    sumDist += np.linalg.norm(path[i]-path[i+1])
    dists.append(sumDist)  
  return dists

def resamplePath(path, samples):
  dists = pathLengthProg(path)
  points = np.linspace(0,dists[-1])
  interpd = []
  for p in range(len(points)):
    ind = 0
    if p == 0:
      interpd.append(path[0])
    elif p==len(points)-1:
      interpd.append(path[-1])
    else:
      for d in range(len(dists)-1):
        if dists[d]<=points[p] and dists[d+1]>points[p]:
          ind = d
      p1norm = np.abs(dists[ind]-p)
      p2norm = np.abs(p-dists[ind+1])
      interpd.append((path[ind]*p1norm + path[ind+1]*p2norm)/(p1norm+p2norm))

  return np.stack(interpd)
      
  
  





class LandscapeProblem():
  def __init__(self, model, batch_size=200):
    self.batch_size = batch_size
    self.pivot_train = np.inf
    self.pivot_test = np.inf
    self.load_dataset()
    self.model = model

  def get_batch(self):
    raise NotImplementedError()

  def dataset_len(self):
    return -1

  def load_dataset(self):
    raise NotImplementedError()

  @partial(jax.jit, static_argnums=(0,))
  def eval_params(self, params, batch, label):
    raise NotImplementedError()

  def train_path(self, 
                 optimizer, 
                 starting_params=None, 
                 epochs=50, 
                 log_name='model', 
                 single_sample=False,
                 save=False,
                 test=True,
                 sample_rate=1,
                 **kwargs):
    params_path = []
    
    params = self.model.init(jax.random.PRNGKey(0), self.get_batch()[0]) if starting_params == None else starting_params
    #params = flax.core.frozen_dict.unfreeze(params)
    #initializer = jax.nn.initializers.kaiming_normal()
    #print(params.keys())
    #for key in params.keys():
      #print(params[key].keys())
      #for subkey in params[key]:
        #print(params[key][subkey].keys())        
        #params[key][subkey]["kernel"] = initializer(jax.random.PRNGKey(42), params[key][subkey]["kernel"].shape, jnp.float32)  
        #params[key][subkey]["kernel"]*=1
    #params = flax.core.frozen_dict.freeze(params)

    opt_state = optimizer.init(params)
    loss_value = -1

    @jax.jit
    def step(params, opt_state, batch, labels):
      loss_value, grads = jax.value_and_grad(self.eval_params)(params, batch, labels)
      updates, opt_state = optimizer.update(grads, opt_state, params)
      params = optax.apply_updates(params, updates)
      return params, opt_state, loss_value

    total_epochs = epochs if not single_sample else 3
    grokometer = 0
    best_test_loss = np.inf
    best_train_loss = np.inf
    for epoch in range(total_epochs):
      if epoch%50==0:
        print('Epoch %d/%d' % (epoch, total_epochs))
      
      train_loss = 0.0
      iters = self.dataset_len()//self.batch_size
      #dataset_iter = tqdm(range(iters))
      for i in range(iters):
        batch, labels = self.get_batch()        
        params, opt_state, loss_value = step(params, opt_state, batch, labels)
        #dataset_iter.set_description('Loss: %f' % loss_value)
        if single_sample:
          params_path.append(params)
        train_loss += loss_value
      train_loss /= iters
      if train_loss<best_train_loss:
        best_train_loss=train_loss


      if not single_sample and epoch%sample_rate==0:
        params_path.append(params)

      if epoch%50==0:
        train_acc = self.accuracy(params, batch, labels)
        if test:
          #print("Testing...")
          test_loss = 0.0
          iters = self.dataset_len(False)//self.batch_size 
          #test_iter = tqdm(range(iters))
          for i in range(iters):
            batch, labels = self.get_batch(False)
            test_value = self.eval_params(params, batch, labels)
            test_loss += test_value            
            #test_iter.set_description('Loss: %f' % loss_value)
          test_loss /= iters
          test_acc = self.accuracy(params, batch, labels)

          if test_loss<best_test_loss:
            best_test_loss=test_loss
        print('Train Loss: %f' % (train_loss))
        print('Train Acc.: %f' % (train_acc))
        print('Best Train Loss: %f' % (best_train_loss))
        if test:
          print('Test Loss: %f' % (test_loss))
          print('Test Acc.: %f' % (test_acc))
          print('Best Test Loss: %f' % (best_test_loss))
          #print(self.accuracy(params, batch, labels))
        print("\n")
      if test_acc>.96 and epoch%sample_rate==0:
        grokometer+=1
        if grokometer == 2:
            break  
    return params_path

class LossVisualizer():
  def flatten_path(self, parameter_path):
    print(len(parameter_path))
    _, unravel = jax.flatten_util.ravel_pytree(parameter_path[0])
    self.unravel = unravel
    params = jax.flatten_util.ravel_pytree(parameter_path.pop(0))[0][None,...]
    print("Flattening Parameter Path...")
    for i in tqdm(range(len(parameter_path))):
      params = np.concatenate((params,jax.flatten_util.ravel_pytree(parameter_path.pop(0))[0][None,...]))
      #print(i)
      #parameter_path[i] = jax.flatten_util.ravel_pytree(parameter_path[i])[0]
    #param_vec = [jax.flatten_util.ravel_pytree(params)[0] for params in parameter_path]
    #param_vec = jnp.stack(param_vec)
    #parameter_path = jnp.stack(parameter_path)
    print("Done!")
    return params
  
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
    pca.fit(np.concatenate((param_vec,-param_vec)))###################
    self.lowdim_fit = pca
    # print(pca.explained_variance_ratio_)
    return pca
  
  def uniform_gap_distribution(self, landscape, parameter_path, samples=100):
    param_dim = self.flatten_path(parameter_path).shape[1]
    rand_projs = jax.random.uniform(jax.random.PRNGKey(0), (samples, param_dim, 2))
    rand_projs = rand_projs / jnp.linalg.norm(rand_projs, axis=1, keepdims=True)
    gaps = []
    for rand_proj in tqdm(rand_projs):
      surface_data = self.process(landscape, 
                                  parameter_path, 
                                  x_vec=rand_proj[:,0], 
                                  y_vec=rand_proj[:,1], 
                                  margin_factor=2e4, 
                                  verbose=False)
      gap = jnp.amin(surface_data['loss_z']) - jnp.amin(surface_data['z'])
      gaps.append(float(gap))
    return gaps
  
  def mill_plot(self, landscape, parameter_path, filenames, **kwargs):
    surface_data = self.process(landscape, parameter_path, **kwargs)

    dir_x_y = "null" #[surface_data['dir_x'].astype(np.float16).tolist(),surface_data['dir_y'].astype(np.float16).tolist()],

    output_train = {
      'path': {
        'bounds': [[float(jnp.amin(surface_data['loss_x'])), float(jnp.amax(surface_data['loss_x']))],
                   [float(jnp.amin(surface_data['loss_y'])), float(jnp.amax(surface_data['loss_y']))],
                   [float(jnp.amin(surface_data['loss_z_train'])), float(jnp.amax(surface_data['loss_z_train']))]],
        'data': jnp.stack([surface_data['loss_x'], surface_data['loss_y'], surface_data['loss_z_train']], axis=1).tolist(),
        'directions': dir_x_y,
        'notes': ['Minimum Path Loss: %f' % jnp.amin(surface_data['loss_z_train'])]
      },
      'surface': {
        'bounds': [[float(jnp.amin(surface_data['x'])), float(jnp.amax(surface_data['x']))],
                   [float(jnp.amin(surface_data['y'])), float(jnp.amax(surface_data['y']))],
                   [float(jnp.amin(surface_data['z_train'])), float(jnp.amax(surface_data['z_train']))]],
        'data': surface_data['z_train'].tolist(),
        'directions': dir_x_y,
        'notes': ['Minimum Sampled Loss: %f' % jnp.amin(surface_data['z_train'])]
      }
    }
    output_test = {
      'path': {
        'bounds': [[float(jnp.amin(surface_data['loss_x'])), float(jnp.amax(surface_data['loss_x']))],
                   [float(jnp.amin(surface_data['loss_y'])), float(jnp.amax(surface_data['loss_y']))],
                   [float(jnp.amin(surface_data['loss_z_test'])), float(jnp.amax(surface_data['loss_z_test']))]],
        'data': jnp.stack([surface_data['loss_x'], surface_data['loss_y'], surface_data['loss_z_test']], axis=1).tolist(),
        'directions': dir_x_y,
        'notes': ['Minimum Path Loss: %f' % jnp.amin(surface_data['loss_z_test'])]
      },
      'surface': {
        'bounds': [[float(jnp.amin(surface_data['x'])), float(jnp.amax(surface_data['x']))],
                   [float(jnp.amin(surface_data['y'])), float(jnp.amax(surface_data['y']))],
                   [float(jnp.amin(surface_data['z_test'])), float(jnp.amax(surface_data['z_test']))]],
        'data': surface_data['z_test'].tolist(),
        'directions': dir_x_y,
        'notes': ['Minimum Sampled Loss: %f' % jnp.amin(surface_data['z_test'])]
      }
    }

    with open(filenames[0], 'w') as output_fp:
      json.dump(output_train, output_fp)
      print('Wrote JSON output to %s.' % filenames[0])
    
    with open(filenames[1], 'w') as output_fp:
      json.dump(output_test, output_fp)
      print('Wrote JSON output to %s.' % filenames[1])

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
              margin_factor=1.25,
              filter_norm=False,
              verbose=True,
              sample_size =256):
    param_vec = self.flatten_path(parameter_path)[len(parameter_path)//2:]
    
    param_vec = resamplePath(param_vec,256)

    self.base_vec = param_vec[-1,:][None,...]
    param_delta = param_vec - self.base_vec
    #print(param_delta[-1])

    self.project(param_delta)

    if filter_norm:
      for i in range(2):
        self.lowdim_fit.components_[i] = self.filter_norm(self.lowdim_fit.components_[i], parameter_path[-1])

    lowdim_path = self.lowdim_fit.transform(param_delta)

    

    amp = margin_factor * np.amax(np.abs(lowdim_path-lowdim_path[-1,:])) if x_range == None else x_range
    #y_amp = margin_factor * np.amax(np.abs(lowdim_path[:,1])) if y_range == None else y_range
    # x_amp = y_amp = margin_factor

    #print(lowdim_path[-1,:])
    #print(lowdim_path[0,:])

    x_mesh, y_mesh = jnp.meshgrid(jnp.linspace(lowdim_path[-1,0]-amp, lowdim_path[-1,0]+amp, resolution), jnp.linspace(lowdim_path[-1,1]-amp, lowdim_path[-1,1]+amp, resolution))

    deltas = jnp.stack([x_mesh, y_mesh], axis=2)
    #print(deltas[50,50,:])
    if verbose: print(deltas.shape)
    
    """ iters = landscape.dataset_len(False)//landscape.batch_size
    param_losses = np.zeros(param_vec.shape[0])
    for i in range(iters):
      batch, label = landscape.get_batch(training)
      param_losses += self.plot_path(landscape, param_vec, batch, label)
    param_losses/=iters

    """

    #Plot paths and surfaces

    dir_x = self.lowdim_fit.components_[0] if x_vec is None else x_vec
    dir_y = self.lowdim_fit.components_[1] if y_vec is None else y_vec
    landscape.batch_size = sample_size
    
    #Training
    landscape.pivot_train= np.inf
    batch, label = landscape.get_batch(training=True)
    param_losses_train = self.plot_path(landscape, param_vec, batch, label)
    print(param_losses_train[-1])
    
    Z_train = jax.lax.map(lambda row_deltas:
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

    if verbose: print(Z_train.shape)

    #Test
    landscape.pivot_test = np.inf
    batch, label = landscape.get_batch(training=False)
    param_losses_test = self.plot_path(landscape, param_vec, batch, label)
    print(param_losses_test[-1])
    
    Z_test = jax.lax.map(lambda row_deltas:
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

    #print(Z[50,50])
    if verbose: print(Z_test.shape)

    return {
        'x': x_mesh,
        'y': y_mesh,
        'z_train': Z_train,
        'z_test': Z_test,
        'loss_x': lowdim_path[:,0],
        'loss_y': lowdim_path[:,1],
        'loss_z_train': param_losses_train,
        'loss_z_test': param_losses_test,
        'dir_x': dir_x,
        'dir_y': dir_y,
        'base_vec': param_vec[-1,:],
        'lowdim_path': lowdim_path
    }
