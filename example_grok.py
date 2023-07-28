from plot import *
from models import *
from statistics import mean 
import torch
import torchvision
from torch.utils.data import DataLoader
import math



train_set = torchvision.datasets.MNIST('./data', train=True, download=True)
test_set = torchvision.datasets.MNIST('./data', train=False, download=True)

train_set_array = train_set.data.numpy()
test_set_array = test_set.data.numpy()

train_labels = torchvision.datasets.MNIST('./data', train=True, download=True)
test_labels = torchvision.datasets.MNIST('./data', train=False, download=True)

train_labels_array = np.zeros((train_set_array.shape[0],10))
train_labels_array[np.arange(train_set_array.shape[0]),train_labels.targets.numpy()-1] = 1

test_labels_array = np.zeros((test_set_array.shape[0],10))
test_labels_array[np.arange(test_set_array.shape[0]),test_labels.targets.numpy()-1] = 1

train_set_array = train_set_array.reshape((train_set_array.shape[0],-1))
test_set_array = test_set_array.reshape((test_set_array.shape[0],-1))

indices = np.arange(train_set_array.shape[0])
np.random.shuffle(indices)


train_set_array = train_set_array[indices][0:1000]
train_labels_array = train_labels_array[indices][0:1000]

test_set_array = test_set_array[0:1000]
test_labels_array = test_labels_array[0:1000]

print("Training set loaded...")
print(train_set_array.shape)
print(train_labels_array.shape)

print("Test set loaded...")
print(test_set_array.shape)
print(test_labels_array.shape)

indices = np.arange(1000)
np.random.shuffle(indices)

class GrokMNIST(LandscapeProblem):
    def get_batch(self,training=True,indices=indices):
        if training:
            if self.pivot_train+self.batch_size>self.dataset_len():
                #print(train_labels_array[[0,100,200,300,400,500]])  
                train_set_array[indices,:] = train_set_array
                train_labels_array[indices,:]  = train_labels_array
                self.pivot_train = 0
            batch = train_set_array[self.pivot_train:self.pivot_train+self.batch_size]
            labels = train_labels_array[self.pivot_train:self.pivot_train+self.batch_size]
            self.pivot_train+=self.batch_size  
        else:
            if self.pivot_test+self.batch_size>self.dataset_len(False):
                indices = np.arange(self.dataset_len(False))
                np.random.shuffle(indices)
                test_set_array[indices] = test_set_array 
                test_labels_array[indices] = test_labels_array
                self.pivot_test = 0
            batch = test_set_array[self.pivot_test:self.pivot_test+self.batch_size]
            labels = test_labels_array[self.pivot_test:self.pivot_test+self.batch_size]
            self.pivot_test+=self.batch_size            
        return batch, labels

    def dataset_len(self,training=True):
        if training:
            return train_set_array.shape[0]
        else:
            return test_set_array.shape[0]
        
    def load_dataset(self):
        pass

    def accuracy(self, params, batch, label):
        #print(batch.shape)
       
        prediction = self.model.apply(params, batch)  
        #print(prediction.shape)
        #print([jnp.argmax(prediction,1)==jnp.argmax(label,1)])     
        return np.mean([jnp.argmax(prediction,1)==jnp.argmax(label,1)])
        

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

#optimizer = optax.adamw(1e-4)
optimizer = optax.sgd(1e-2,momentum=.9,nesterov=True)
parameter_path = landscape.train_path(optimizer, epochs=30000,sample_rate=50)

vis = LossVisualizer()

vis.mill_plot(landscape, parameter_path, filenames=['data_mnist_train.json','data_mnist_test.json'], resolution=101, margin_factor=1.5)
#vis.mill_plot(landscape, parameter_path, filename='data_mnist_test.json', new_fit=False, resolution=101, margin_factor=2,training=False)

# fig = vis.plotly_plot(landscape, parameter_path, resolution=101, margin_factor=100, sample_idx=3000)
# fig.write_html('data_mnist_test.html')

# fig = vis.plotly_plot(landscape, parameter_path, resolution=101, margin_factor=100, sample_idx=0)
# fig.write_html('data_mnist_train.html')
