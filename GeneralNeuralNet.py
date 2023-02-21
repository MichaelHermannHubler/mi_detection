from ds_CPSC2018 import *
import Utils as utls

import torch.nn as nn
import seaborn as sn
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import numpy as np

from torch.nn import CrossEntropyLoss

from sklearn.metrics import confusion_matrix

class CNN(nn.Module):
    def __init__(self, input_dim):
        super(CNN, self).__init__()
        
        self.conv1 = utls.ConvolutionBlock(in_channels=12, out_channels=32)
        self.conv2 = utls.ConvolutionBlock(in_channels=32, out_channels=32)
        self.conv3 = utls.ConvolutionBlock(in_channels=32, out_channels=32)
        self.conv4 = utls.ConvolutionBlock(in_channels=32, out_channels=32)
        self.conv5 = utls.ConvolutionBlock(in_channels=32, out_channels=32)
        self.conv6 = utls.ConvolutionBlock(in_channels=32, out_channels=32)
        self.conv7 = utls.ConvolutionBlock(in_channels=32, out_channels=32)
        self.conv8 = utls.ConvolutionBlock(in_channels=32, out_channels=32) 
        
        # fully connected layer, output 2 classes
        #self.out = nn.Linear(1216, 2) # 8
        
        self.lin1 = nn.Sequential(      
            nn.Dropout(p=0.2),
            nn.Linear(288, 512),
            nn.ReLU(),
        )    
        self.lin2 = nn.Sequential(      
            nn.Dropout(p=0.2),
            nn.Linear(512, 128),
            nn.ReLU(),
        )    
        self.out = nn.Linear(128, 2)

        self.is_conv = True
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)

        # flatten the output of conv2 to (batch_size, 32 * 7 * 7)
        x = x.view(x.size(0), -1)    
        x = self.lin1(x)
        x = self.lin2(x)

        output = self.out(x)
        return output 


def train(model, train_loader, optimizer, loss_fun, device, epoch):
    # TODO adapt code below
    model.train()
    
    n_batches = len(train_loader)
    for i, (signal, _, _, targets) in enumerate(train_loader):
        
        signal = signal.to(device)
        targets = targets.to(device)

        # get batch size
        bs = signal.shape[0]
        # print(i, bs)
            
        # fully connected model: we need to flatten the signals

        x = signal.view(bs,-1) if not model.is_conv else signal.view(bs, 12, 5000)
            
        # signal to device
        x = x.to(device)
            
        # zero grads
        optimizer.zero_grad()
            
        # forward pass
        out = model(x)
            
        # calc loss and gradients
        loss = loss_fun(out, targets).mean()
        loss.backward()
            
        # update
        optimizer.step()
    return loss.item()


def test(model, train_loader, optimizer, loss_fun, device, epoch):
    # TODO: adapt code beolow
    model.train()
    
    n_batches = len(train_loader)
    for i, (signal, _, _, target) in enumerate(train_loader):
        # get batch size
        bs = signal.shape[0]
            
        # fully connected model: we need to flatten the signals
        x = signal.view(bs, -1) if not model.is_conv else signal.view(bs, 12, 5000)
            
        # signal to device
        x = x.to(device)
            
        # zero grads
        optimizer.zero_grad()
            
        # forward pass
        out = model(x)
            
        # calc loss and gradients
        loss = loss_fun(out, target).mean()
        loss.backward()
            
        # update
        optimizer.step()
    return loss.item()

def main():
    num_epochs = 100
    model = CNN((12,7500))

    print(model)

    optimizer = optim.Adam(params=model.parameters(),lr=0.0005)
    ce_loss = CrossEntropyLoss()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    tr_loss = []
    tr_acc = []
    ev_loss = []
    ev_acc = []
    for epoch in range(num_epochs):
        loss = train(model, train_loader, optimizer, ce_loss, device, epoch)
        tr_loss.append(loss)
                
        # calculate accuracy
        model.eval()
        N = 2000
        x, _, _, label = dataset[:N] 
        x = x.view(N, 12, 5000) if model.is_conv else  x.view(N,-1) 

        x = x.to(device)
        label = label.to(device)
        out = model(x)
        acc_ = (out.argmax(-1) == label).float().sum()/len(label)
        acc_ = acc_.to('cpu')
        tr_acc.append(acc_)


        x, _, _, label = dataset[:N] 
        x = x.view(N, 12, 5000) if model.is_conv else  x.view(N,-1)
        model.eval()
        
        x = x.to(device)
        label = label.to(device)
        out = model(x)

        acc_ = (out.argmax(-1) == label).float().sum()/len(label)
        acc_ = acc_.to('cpu')

        ev_acc.append(acc_)
        
        
        print(f'epoch [{epoch+1}/{num_epochs}]: train loss = {loss:.8f}, train acc = {tr_acc[-1]:.5f}, val acc = {ev_acc[-1]:.5f}')
    

    plt.plot(tr_loss, label='train loss')
    plt.legend()
    plt.show()
    
    plt.plot(tr_acc, label='train accuracy')
    plt.plot(ev_acc, label='eval accuracy')
    plt.title('acc')
    plt.legend()
    plt.show()

    return model

dataset = CPSC2018Dataset()
train_loader, test_loader, val_loader = dataset.get_Loaders()
path = "G:\\Projects\\MA"
model_version = 1


if os.path.exists(os.path.join(path, f'model{model_version}.chpt')):
    model = CNN((12,7500))
    model.load_state_dict(torch.load(os.path.join(path, f'model{model_version}.chpt')))
else:
    model = main()
    torch.save(model.state_dict(), os.path.join(path, f'model{model_version}.chpt'))

model.eval()
model.to('cpu')

def data_test(model, signal, ytrue):
    pred = model(signal.view(signal.shape[0], 12, 5000))
    pred = torch.argmax(pred)
    if pred != ytrue:
        print('Prediction: ', pred, 'Real: ', ytrue)

y_pred = []
y_true = []

# iterate over test data
for inputs, _, _, labels in val_loader:
    output = model(inputs.view(inputs.shape[0], 12, 5000)) # Feed Network

    output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
    y_pred.extend(output) # Save Prediction
    
    labels = labels.data.cpu().numpy()
    y_true.extend(labels) # Save Truth

classes = ('Normal', 'Abnormal')
cf_matrix = confusion_matrix(y_true, y_pred)
df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix), index = [i for i in classes],
                     columns = [i for i in classes])
plt.figure(figsize = (12,7))
sn.heatmap(df_cm, annot=True)
plt.show()