from ds_PTBXL import *
import Utils as utls

import torch.nn as nn
import seaborn as sn
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import numpy as np

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm
from copy import deepcopy

import torch.nn as nn
import torch
import pickle
from SpecialistNeuralNetworkBase import CNN, FinetunedCNN

# create the new model
class UpscalingCNN(nn.Module):
    def __init__(self, specializationModel = FinetunedCNN(CNN())):
        super(UpscalingCNN, self).__init__()
        self.upscaler = nn.Sequential(         
            nn.Conv1d(in_channels=6, out_channels=12, kernel_size=50, stride=1, padding=25),                
            nn.LeakyReLU(),
            nn.Dropout1d(p=0.1),
            nn.BatchNorm1d(12),
        )

        self.specializationModel = specializationModel

        self.is_conv = True

    def forward(self, x):
        x = self.upscaler(x)
        x = self.specializationModel(x)

        return x

def train(model, train_loader, optimizer, loss_fun, device):
    model.train()
    run_loss = 0
    
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
        run_loss += loss.item()

    #print(loss)
    return run_loss / len(train_loader)

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
    generalizationModel = CNN()
    spezialisationModel = FinetunedCNN(generalizationModel)
    spezialisationModel.load_state_dict(torch.load("G:\\Projects\\MA\\models\\" + 'specbase_modelfinalBase.chpt'))

    model = UpscalingCNN(spezialisationModel)

    return model

def main():
    num_epochs = 150
    num_folds = 5
    depth = 'final_Basev2'
    lr = 0.005
    
    best_acc = 0
    epoch_since_best = 0
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
        epoch_since_best_run = 0

        for epoch in range(num_epochs):
            loss = train(model, train_loader, optimizer, loss_fun, device)
            tr_loss.append(loss)
            tr_acc.append(test(model, train_loader, device))
            te_acc.append(test(model, test_loader, device))

            epoch_since_best_run += 1
            epoch_since_best += 1

            if te_acc[-1] > best_run_acc:
                best_run_acc = te_acc[-1]
                epoch_since_best_run = 0
            
            if te_acc[-1] > best_acc:
                best_acc = te_acc[-1]
                best_state = deepcopy(model.state_dict())
                epoch_since_best = 0
            
            print(f'epoch [{epoch+1}/{num_epochs}]: train loss = {loss:.8f}, train acc = {tr_acc[-1]:.5f}, test acc = {te_acc[-1]:.5f}, best run acc = {best_run_acc:.5f}({epoch_since_best_run}), best acc = {best_acc:.5f}({epoch_since_best})')
            
            if epoch_since_best_run > 20:
                print(f'early stopping at epoch {epoch}')
                break        
        plt.plot(tr_loss, label='train loss')
        plt.legend()
        plt.savefig(f'G:\Projects\MA\images\\UNN\{depth}_Layers_{fold + 1}_Loss_lr{lr}.png')
        plt.clf()
        
        plt.plot(tr_acc, label='train accuracy')
        plt.plot(te_acc, label='test accuracy')
        plt.legend()
        plt.savefig(f'G:\Projects\MA\images\\UNN\{depth}_Layers_{fold + 1}_Acc_lr{lr}.png')
        plt.clf()

        with open(f'G:\Projects\MA\\variables\\UNN\{depth}_Layers_{fold + 1}_train_loss.pkl', 'wb') as f:
            pickle.dump(tr_loss, f)
            
        with open(f'G:\Projects\MA\\variables\\UNN\{depth}_Layers_{fold + 1}_train_acc.pkl', 'wb') as f:
            pickle.dump(tr_acc, f)
            
        with open(f'G:\Projects\MA\\variables\\UNN\{depth}_Layers_{fold + 1}_test_acc.pkl', 'wb') as f:
            pickle.dump(te_acc, f)

    print(f'Best test acc = {best_acc:.5f}')
    model.load_state_dict(best_state)

    return model

def loadLayer(state_dict, layerName, layer):
    layer.weight = torch.nn.Parameter(state_dict[layerName + 'weight'])
    layer.bias = torch.nn.Parameter(state_dict[layerName + 'bias'])


if __name__=="__main__":
    train_dataset = PTBXLDataset(labels = ['NORM'], folds=range(6,8), leads=range(6))
    test_dataset = PTBXLDataset(labels = ['NORM'], folds=[9], leads=range(6))
    val_dataset = PTBXLDataset(labels = ['NORM'], folds=[10], leads=range(6))

    train_loader = train_dataset.get_full_loader()
    test_loader = test_dataset.get_full_loader()
    val_loader = val_dataset.get_full_loader()

    path = "G:\\Projects\\MA\\models\\"
    model_version = 'finalBasev2'

    recalculate = False

    if os.path.exists(os.path.join(path, f'upscalebase_model{model_version}.chpt')) and not recalculate:
        model = UpscalingCNN()
        model.load_state_dict(torch.load(os.path.join(path, f'upscalebase_model{model_version}.chpt')))
    else:
        model = main()
        torch.save(model.state_dict(), os.path.join(path, f'upscalebase_model{model_version}.chpt'))
        
        model = UpscalingCNN()
        model.load_state_dict(torch.load(os.path.join(path, f'upscalebase_model{model_version}.chpt')))

    model.eval()
    model.to('cpu')

    y_pred = []
    y_true = []

    # iterate over test data
    for inputs, _, _, labels in val_loader:
        output = model(inputs.view(inputs.shape[0], 6, 4000)) # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output) # Save Prediction
        
        labels = labels['NORM'].cpu().numpy()
        y_true.extend(labels) # Save Truth

    print('Accuracy: ', accuracy_score(y_true, y_pred))
    print('Precision: ', precision_score(y_true, y_pred))
    print('Recall: ', recall_score(y_true, y_pred))
    print('F1 Score: ', f1_score(y_true, y_pred))

    classes = ('MI', 'non-MI')
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix), index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.show()