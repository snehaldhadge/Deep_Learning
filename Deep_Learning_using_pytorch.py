
import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms
from torchsummary import summary

import matplotlib.pyplot as plt

# like to apply some trasnforms to data
# toTensor scales data from e to use this ToTensor method 
# which scales our pixels down from zero to 255 integers down
#  to zero to one float values. [0,255] to [0,1]
train = MNIST('data',train=True,transform = transforms.ToTensor(),download=True)
test = MNIST('data',train=False,transform=transforms.ToTensor())

train.data.shape
test.data.shape

train.data[0] # not scaled
# transofrmation will happen later

plt.imshow(train.data[0].numpy().squeeze(),cmap='gray_r')

# labels are in target
train.targets[0:10]

# Batch Data
# here we apply the transform
train_loader =torch.utils.data.DataLoader(train,batch_size=128,shuffle=True)
test_loader = torch.utils.data.DataLoader(test,batch_size=128)

X_sample,y_sample = iter(train_loader).next()
X_sample.shape

X_sample[0]

# reshapes tensor
# (here we have 60000 samples and batch_size of 128 so 60000/128 gives 486.75 so very few samples in last batch
# to avoid such error we use shape )
X_flat_sample = X_sample.view(X_sample.shape[0],-1)
X_flat_sample.shape
# %%
# Neural network Architecture
n_input = 784
n_dense = 64
n_out = 10

model = nn.Sequential(
    nn.Linear(n_input,n_dense),#hidden layer
    nn.ReLU(),# activation layer
    nn.Linear(n_dense,n_out)
)
summary(model,(1,n_input))

# %%
# Configure training hyper paramaters

cost_fxn = nn.CrossEntropyLoss() # includes softmax activation
optimizer = torch.optim.Adam(model.parameters())

# There is no metrics like accuracy in pytorch so need to write a function

def accuracy_pct(pred_y,true_y):
    _,prediction = torch.max(pred_y,1)
    correct = (prediction==true_y).sum().item()
    return (correct/true_y.shape[0]) * 100.0


n_batches = len(train_loader)
n_batches
n_epochs=10

print('Training for {} epochs. \n'.format(n_epochs))

for epoch in range(n_epochs):
    avg_cost = 0.0
    avg_accuracy = 0.0

    for i ,(X,y) in enumerate(train_loader): # enumerate() provides count of iterations  
    
    # forward propagation:
        X_flat = X.view(X.shape[0], -1)
        y_hat = model(X_flat)
        cost = cost_fxn(y_hat, y)
        avg_cost += cost / n_batches
        
        # backprop and optimization via gradient descent: 
        optimizer.zero_grad() # set gradients to zero; .backward() accumulates them in buffers
        cost.backward()
        optimizer.step()
        
        # calculate accuracy metric:
        accuracy = accuracy_pct(y_hat, y)
        avg_accuracy += accuracy / n_batches
        
        if (i + 1) % 100 == 0:
            print('Step {}'.format(i + 1))
    
    print('Epoch {}/{} complete: Cost: {:.3f}, Accuracy: {:.1f}% \n'
        .format(epoch + 1, n_epochs, avg_cost, avg_accuracy)) 

print('Training complete.')


# Test Model
n_test_batches = len(test_loader)
n_test_batches

model.eval() # disables dropout and batch norm



with torch.no_grad(): # disables autograd, reducing memory consumption
  
  avg_test_cost = 0.0
  avg_test_acc = 0.0
  
  for X,y in test_loader:
    
    # make predictions: 
    X_flat = X.view(X.shape[0], -1)
    y_hat = model(X_flat)
    
    # calculate cost: 
    cost = cost_fxn(y_hat, y)
    avg_test_cost += cost / n_test_batches
    
    # calculate accuracy:
    test_accuracy = accuracy_pct(y_hat, y)
    avg_test_acc += test_accuracy / n_test_batches

print('Test cost: {:.3f}, Test accuracy: {:.1f}%'.format(avg_test_cost, avg_test_acc))


# Test cost: 0.087, Test accuracy: 97.6%
# 


