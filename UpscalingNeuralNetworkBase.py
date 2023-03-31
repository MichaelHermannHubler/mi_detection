from ds_PTBXL import *
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
from tqdm import tqdm
from copy import deepcopy

import torch.nn as nn
import torch
import pickle

# create the new model
class UpscaleCNN(nn.Module):
    def __init__(self):
        super(UpscaleCNN, self).__init__()

        self.upscaler = nn.Sequential(         
            nn.Conv1d(in_channels=6, out_channels=12, kernel_size=50, stride=1, padding=25),                
            nn.LeakyReLU(),
            nn.Dropout1d(p=0.1),
            nn.BatchNorm1d(12),
        )

        # initialize the same architecture as the existing CNN model 
        self.conv1 = utls.ConvolutionBlock(in_channels=12, out_channels=24)
        self.conv2 = utls.ConvolutionBlock(in_channels=24, out_channels=24)
        self.conv3 = utls.ConvolutionBlock(in_channels=24, out_channels=24)
        self.conv4 = utls.ConvolutionBlock(in_channels=24, out_channels=36)
        self.conv5 = utls.ConvolutionBlock(in_channels=36, out_channels=36)
        self.conv6 = utls.ConvolutionBlock(in_channels=36, out_channels=36)
        self.conv7 = utls.ConvolutionBlock(in_channels=36, out_channels=48)
        self.conv8 = utls.ConvolutionBlock(in_channels=48, out_channels=48)
        self.conv9 = utls.ConvolutionBlock(in_channels=48, out_channels=48)
        self.conv10 = utls.ConvolutionBlock(in_channels=48, out_channels=48) 

        
        self.lin1 = nn.Sequential(      
            nn.Dropout(p=0.2),
            nn.Linear(96, 32),
            nn.ReLU(),
        )        
        self.lin2 = nn.Sequential(      
            nn.Dropout(p=0.2),
            nn.Linear(32, 32),
            nn.ReLU(),
        )   

        self.norm = utls.OutputBlock(in_channels=32, out_channels=2)

        self.is_conv = True 

    def forward(self, x):
        x = self.upscaler(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.conv7(x)
        x = self.conv8(x)
        x = self.conv9(x)
        x = self.conv10(x)

        x = x.view(x.size(0), -1)

        x = self.lin1(x)
        x = self.lin2(x)
        # x = self.lin3(x)
        
        return self.norm(x)

def train(model, train_loader, optimizer, loss_fun, device):
    model.train()
    
    for i, (signal, _, _, targets) in enumerate(tqdm(train_loader, desc='Train')):
        signal = signal.to(device)
        targets = targets['NORM'].to(device)

        # get batch size
        bs = signal.shape[0]
            
        # fully connected model: we need to flatten the signals
        x = signal.view(bs,-1) if not model.is_conv else signal.view(bs, 6, 4000)
            
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

    #print(loss)
    return loss.item()

def test(model, test_loader, device):
    model.eval()    

    accs = []
    for i, (signal, _, _, targets) in enumerate(tqdm(test_loader, desc='Test')):
        signal = signal.to(device)
        targets = targets['NORM'].to(device)
        
        bs = signal.shape[0]

        x = signal.view(bs,-1) if not model.is_conv else signal.view(bs, 6, 4000)
        x = x.to(device)

        out = model(x)
        acc_ = (out.argmax(-1) == targets).float().sum()/len(targets)
        acc_ = acc_.to('cpu')

        accs.append(acc_)

    acc = sum(accs)/len(accs)

    return acc

def criterion(loss_func, outputs, labels, device):
  losses = 0
  for i, key in enumerate(outputs):
    losses += loss_func(outputs[key], labels[key].to(device))
  return losses

def loadModel():
    model = UpscaleCNN()

    # load the weights from the file
    state_dict = torch.load("G:\\Projects\\MA\\models\\" + 'specbase_model1.chpt')

    utls.loadConvolutionBlockLayer(state_dict, 'conv1', model.conv1)
    utls.loadConvolutionBlockLayer(state_dict, 'conv2', model.conv2)
    utls.loadConvolutionBlockLayer(state_dict, 'conv3', model.conv3)
    utls.loadConvolutionBlockLayer(state_dict, 'conv4', model.conv4)
    utls.loadConvolutionBlockLayer(state_dict, 'conv5', model.conv5)
    utls.loadConvolutionBlockLayer(state_dict, 'conv6', model.conv6)
    utls.loadConvolutionBlockLayer(state_dict, 'conv7', model.conv7)
    utls.loadConvolutionBlockLayer(state_dict, 'conv8', model.conv8)
    utls.loadConvolutionBlockLayer(state_dict, 'conv9', model.conv9)
    utls.loadConvolutionBlockLayer(state_dict, 'conv10', model.conv10)
    
    loadLayer(state_dict, 'lin1.1.', model.lin1[1])
    loadLayer(state_dict, 'lin2.1.', model.lin2[1])
    loadLayer(state_dict, 'norm.out.1.', model.norm.out[1])

    # freeze layers
    # for name, param in model.named_parameters():
    #     if 'conv1' in name or 'conv2' in name or 'conv3' in name or 'conv4' in name or 'conv5' in name:
    #         param.requiresGrad = False

    return model

def main():
    num_epochs = 150
    num_folds = 10
    depth = 10.1
    lr = 0.005
    
    best_acc = 0
    best_state = []

    for fold in range(num_folds):
        print(f'Fold [{fold + 1}/{num_folds}]:')
        model = loadModel()

        optimizer = optim.Adam(params=model.parameters(), lr=lr)
        loss_fun = nn.CrossEntropyLoss()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        tr_loss = []
        tr_acc = []
        te_acc = []
        best_run_acc = 0
        for epoch in range(num_epochs):
            loss = train(model, train_loader, optimizer, loss_fun, device)
            tr_loss.append(loss)
            tr_acc.append(test(model, train_loader, device))
            te_acc.append(test(model, test_loader, device))

            if te_acc[-1] > best_run_acc:
                best_run_acc = te_acc[-1]
            
            if te_acc[-1] > best_acc:
                best_acc = te_acc[-1]
                best_state = deepcopy(model.state_dict())
            
            print(f'epoch [{epoch+1}/{num_epochs}]: train loss = {loss:.8f}, train acc = {tr_acc[-1]:.5f}, test acc = {te_acc[-1]:.5f}, best run acc = {best_run_acc:.5f}, best acc = {best_acc:.5f}')
        
        plt.plot(tr_loss, label='train loss')
        plt.legend()
        plt.savefig(f'G:\Projects\MA\images\\UNNB\{depth}_Layers_{fold + 1}_Loss_lr{lr}.png')
        plt.clf()
        
        plt.plot(tr_acc, label='train accuracy')
        plt.plot(te_acc, label='test accuracy')
        plt.legend()
        plt.savefig(f'G:\Projects\MA\images\\UNNB\{depth}_Layers_{fold + 1}_Acc_lr{lr}.png')
        plt.clf()

        with open(f'G:\Projects\MA\\variables\\UNNB\{depth}_Layers_{fold + 1}_train_loss.pkl', 'wb') as f:
            pickle.dump(tr_loss, f)
            
        with open(f'G:\Projects\MA\\variables\\UNNB\{depth}_Layers_{fold + 1}_train_acc.pkl', 'wb') as f:
            pickle.dump(tr_acc, f)
            
        with open(f'G:\Projects\MA\\variables\\UNNB\{depth}_Layers_{fold + 1}_test_acc.pkl', 'wb') as f:
            pickle.dump(te_acc, f)

    print(f'Best test acc = {best_acc:.5f}')
    model.load_state_dict(best_state)

    return model

def loadLayer(state_dict, layerName, layer):
    layer.weight = torch.nn.Parameter(state_dict[layerName + 'weight'])
    layer.bias = torch.nn.Parameter(state_dict[layerName + 'bias'])


if __name__=="__main__":
    #dataset = PTBXLDataset(labels = ['NORM'], leads=range(6))
    train_dataset = PTBXLDataset(labels = ['NORM'], folds=range(6,8), leads=range(6))
    test_dataset = PTBXLDataset(labels = ['NORM'], folds=[9], leads=range(6))
    val_dataset = PTBXLDataset(labels = ['NORM'], folds=[10], leads=range(6))

    # train_data, _ = train_test_split(train_dataset, test_size=0.00001, random_state=42)
    # test_data, _ = train_test_split(train_dataset, test_size=0.00001, random_state=42)
    # val_data, _ = train_test_split(train_dataset, test_size=0.00001, random_state=42)

    # train_loader = DataLoader(dataset=train_data, batch_size=32, shuffle=True)
    # test_loader = DataLoader(dataset=test_data, batch_size=32, shuffle=True)
    # val_loader = DataLoader(dataset=val_data, batch_size=32, shuffle=True)

    train_loader = train_dataset.get_full_loader()
    test_loader = test_dataset.get_full_loader()
    val_loader = val_dataset.get_full_loader()

    path = "G:\\Projects\\MA\\models\\"
    model_version = 10.1

    recalculate = True

    if os.path.exists(os.path.join(path, f'upscalebase_model{model_version}.chpt')) and not recalculate:
        model = UpscaleCNN()
        model.load_state_dict(torch.load(os.path.join(path, f'upscalebase_model{model_version}.chpt')))
    else:
        model = main()
        torch.save(model.state_dict(), os.path.join(path, f'upscalebase_model{model_version}.chpt'))
        
        model = UpscaleCNN()
        model.load_state_dict(torch.load(os.path.join(path, f'upscalebase_model{model_version}.chpt')))

    model.eval()
    model.to('cpu')

    y_pred = []
    y_true = []

    # iterate over test data
    for inputs, _, _, labels in val_loader:
        output = model(inputs.view(inputs.shape[0], 12, 4000)) # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output) # Save Prediction
        
        labels = labels['NORM'].cpu().numpy()
        y_true.extend(labels) # Save Truth

    for inputs, _, _, labels in val_loader:
        print(labels['NORM'])
        break

    classes = ('Normal', 'Abnormal')
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix), index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.show()