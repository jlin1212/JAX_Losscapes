from plot import *
from models import *
from statistics import mean 
import torch
import torchvision
from torch.utils.data import DataLoader
import math

#Load Data
print("\n")
print("Loading data...\n") 

train_set = torchvision.datasets.MNIST('./data', train=True, download=True)
test_set = torchvision.datasets.MNIST('./data', train=False, download=True)

train_set_array = train_set.data.numpy()
test_set_array = test_set.data.numpy()

train_labels = torchvision.datasets.MNIST('./data', train=True, download=True)
test_labels = torchvision.datasets.MNIST('./data', train=False, download=True)

#Flatten Images

train_set_array = train_set_array.reshape((train_set_array.shape[0],-1))
test_set_array = test_set_array.reshape((test_set_array.shape[0],-1))

#Categorical to One-Hot

train_labels_array = np.zeros((train_set_array.shape[0],10))
train_labels_array[np.arange(train_set_array.shape[0]),train_labels.targets.numpy()-1] = 1

test_labels_array = np.zeros((test_set_array.shape[0],10))
test_labels_array[np.arange(test_set_array.shape[0]),test_labels.targets.numpy()-1] = 1

#Limit Data 

train_set_array = train_set_array[0:1000]
train_labels_array = train_labels_array[0:1000]

test_set_array = test_set_array[0:1000]
test_labels_array = test_labels_array[0:1000]

#Report

print("Training set loaded:")
print(train_set_array.shape)
print(train_labels_array.shape)

print("Test set loaded:")
print(test_set_array.shape)
print(test_labels_array.shape)
print("\n") 

#Define Grok/MNIST Subclass of LandscapeProblem

class GrokMNIST(LandscapeProblem):
    def get_batch(self,training=True):
        """Samples batches of size self.batch_size without replacement. Reshuffles at end of epoch."""  

        if training: #Get Training Data 
            if self.pivot_train+self.batch_size>self.dataset_len(): #End of Epoch
                #Reshuffle Data 
                indices = np.arange(train_set_array.shape[0])
                np.random.shuffle(indices)
                ordered = np.arange(train_set_array.shape[0])
                #Notes: "ordered" required to avoid variable scope error.
                #       "train_set_array[indices] = train_set_array" results in np.shuffle() glitch
                train_set_array[ordered] = train_set_array[indices]
                train_labels_array[ordered]  = train_labels_array[indices]
                #Reset Pivot
                self.pivot_train = 0     
            #Get Batches             
            batch = train_set_array[self.pivot_train:self.pivot_train+self.batch_size]
            labels = train_labels_array[self.pivot_train:self.pivot_train+self.batch_size]  
            #Increment Pivot          
            self.pivot_train+=self.batch_size  
            
        else: #Get Test Data 
            if self.pivot_test+self.batch_size>self.dataset_len(False): #End of Epoch
                #Reshuffle Data
                indices = np.arange(test_set_array.shape[0])
                np.random.shuffle(indices)
                ordered = np.arange(test_set_array.shape[0])
                #Notes: "ordered" required to avoid variable scope error.
                #       "train_set_array[indices] = train_set_array" results in np.shuffle() glitch
                test_set_array[ordered] = test_set_array[indices]
                test_labels_array[ordered]  = test_labels_array[indices]
                #Reset Pivot
                self.pivot_test = 0  
            #Get Batches              
            batch = test_set_array[self.pivot_test:self.pivot_test+self.batch_size]
            labels = test_labels_array[self.pivot_test:self.pivot_test+self.batch_size]
            #Increment Pivot
            self.pivot_test+=self.batch_size            
        return batch, labels

    
    def dataset_len(self,training=True):
        """Returns length of training or test set."""  
        if training:
            return train_set_array.shape[0]
        else:
            return test_set_array.shape[0]
        
    def load_dataset(self):
        pass

    def accuracy(self, params, batch, labels):
        """Returns accuracy of [params] on [batch] given [labels].""" 
        prediction = self.model.apply(params, batch)  
        return jnp.mean(jnp.argmax(prediction,1)==jnp.argmax(labels,1))        

    def eval_params(self, params, batch, labels):
        """Returns loss of [params] on [batch] given [labels].""" 
        prediction = self.model.apply(params, batch)        
        return jnp.mean(optax.l2_loss(jnp.squeeze(prediction), labels))
    
class MLP(nn.Module):
    """A Simple Multi-Layer Perceptron (3 Hiddden Layers, 200 Neurons/Layer)."""  
    @nn.compact
    def __call__(self, x):
      #x = nn.Dense(784)(x)
      #x = nn.relu(x)
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
parameter_path = landscape.train_path(optimizer, epochs=10000,sample_rate=25,test_loss_stop=.95,test_loss_strikes=3)

vis = LossVisualizer()

vis.mill_plot(landscape, parameter_path, filenames=['data_mnist_train.json','data_mnist_test.json'], resolution=101, 
              margin_factor=1.5, truncate_pct = 0, drop_first=1, accuracy=False)

# fig = vis.plotly_plot(landscape, parameter_path, resolution=101, margin_factor=100, sample_idx=3000)
# fig.write_html('data_mnist_test.html')

# fig = vis.plotly_plot(landscape, parameter_path, resolution=101, margin_factor=100, sample_idx=0)
# fig.write_html('data_mnist_train.html')
