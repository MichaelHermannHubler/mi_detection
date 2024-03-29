from ds_General import *
import Utils as utls

import torch.nn as nn
import seaborn as sn
import pandas as pd
import torch.optim as optim
import matplotlib.pyplot as plt
import os
import numpy as np

from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from copy import deepcopy

from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import pickle


torch.set_printoptions(precision=20)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = utls.ConvolutionBlock(in_channels=12, out_channels=24, kernel_size=102)
        self.conv2 = utls.ConvolutionBlock(in_channels=24, out_channels=24, kernel_size=24)
        self.conv3 = utls.ConvolutionBlock(in_channels=24, out_channels=24, kernel_size=11)
        self.conv4 = utls.ConvolutionBlock(in_channels=24, out_channels=36, kernel_size=9)
        self.conv5 = utls.ConvolutionBlock(in_channels=36, out_channels=36)
        self.conv6 = utls.ConvolutionBlock(in_channels=36, out_channels=36)
        self.conv7 = utls.ConvolutionBlock(in_channels=36, out_channels=48)
        self.conv8 = utls.ConvolutionBlock(in_channels=48, out_channels=48)
        self.conv9 = utls.ConvolutionBlock(in_channels=48, out_channels=48)
        self.conv10 = utls.ConvolutionBlock(in_channels=48, out_channels=48)
                
        self.lin1 = nn.Sequential(    
            nn.Linear(336, 128),
            nn.Dropout(p=0.6),
            nn.LeakyReLU(),
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
        x = self.conv9(x)
        x = self.conv10(x)

        # flatten the output of conv
        x = x.view(x.size(0), -1)    
        x = self.lin1(x)

        output = self.out(x)
        return output 

def train(model, train_loader, optimizer, loss_fun, device):
    model.train()
    run_loss = 0
    
    for i, (signal, _, _, targets) in enumerate(tqdm(train_loader, desc='Train')):
        signal = signal.to(device)
        targets = targets.to(device)

        # get batch size
        bs = signal.shape[0]
            
        # fully connected model: we need to flatten the signals
        x = signal.view(bs,-1) if not model.is_conv else signal.view(bs, 12, 4000)
            
        # signal to device
        x = x.to(device)
            
        # zero grads
        optimizer.zero_grad()
            
        # forward pass
        out = model(x)
            
        # calc loss and gradients

        loss = loss_fun(out, targets).mean()
        loss.backward()
        run_loss += loss.item()
            
        # update
        optimizer.step()

    return run_loss / len(train_loader)

def test(model, test_loader, device):
    model.eval()    

    accs = []
    for i, (signal, _, _, targets) in enumerate(tqdm(test_loader, desc='Test')):
        signal = signal.to(device)
        targets = targets.to(device)
        
        bs = signal.shape[0]

        x = signal.view(bs,-1) if not model.is_conv else signal.view(bs, 12, 4000)
        x = x.to(device)

        out = model(x)
        acc_ = (out.argmax(-1) == targets).float().sum()/len(targets)
        acc_ = acc_.to('cpu')

        accs.append(acc_)

    acc = sum(accs)/len(accs)

    return acc

def main():
    num_epochs = 150
    num_folds = 10
    depth = '10.f'
    
    best_acc = 0
    epoch_since_best = 0
    best_state = []

    for fold in range(num_folds):
        print(f'Fold [{fold + 1}/{num_folds}]:')
        model = CNN()

        lr = 1e-3

        optimizer = optim.Adam(params=model.parameters(),lr=lr)
        ce_loss = CrossEntropyLoss()
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        tr_loss = []
        tr_acc = []
        te_acc = []
        best_run_acc = 0
        epoch_since_best_run = 0

        for epoch in range(num_epochs):
            loss = train(model, train_loader, optimizer, ce_loss, device)
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
        plt.savefig(f'G:\Projects\MA\images\GNN\{depth}_Layers_{fold + 1}_Loss_lr{lr}.png')
        plt.clf()
        
        plt.plot(tr_acc, label='train accuracy')
        plt.plot(te_acc, label='test accuracy')
        plt.legend()
        plt.savefig(f'G:\Projects\MA\images\GNN\{depth}_Layers_{fold + 1}_Acc_lr{lr}.png')
        plt.clf()

        with open(f'G:\Projects\MA\\variables\GNN\{depth}_Layers_{fold + 1}_train_loss.pkl', 'wb') as f:
            pickle.dump(tr_loss, f)
            
        with open(f'G:\Projects\MA\\variables\GNN\{depth}_Layers_{fold + 1}_train_acc.pkl', 'wb') as f:
            pickle.dump(tr_acc, f)
            
        with open(f'G:\Projects\MA\\variables\GNN\{depth}_Layers_{fold + 1}_test_acc.pkl', 'wb') as f:
            pickle.dump(te_acc, f)

    print(f'Best test acc = {best_acc:.5f}')
    model.load_state_dict(best_state)

    return model

if __name__=="__main__":
    dataset = GeneralDataset()
    train_loader, test_loader, val_loader = dataset.get_Loaders()
    path = "G:\\Projects\\MA\\models\\"
    model_version = 'final'


    if os.path.exists(os.path.join(path, f'model{model_version}.chpt')):
        model = CNN()
        model.load_state_dict(torch.load(os.path.join(path, f'model{model_version}.chpt')))
    else:    
        model = main()
        torch.save(model.state_dict(), os.path.join(path, f'model{model_version}.chpt'))

    model.eval()
    model.to('cpu')

    def data_test(model, signal, ytrue):
        pred = model(signal.view(signal.shape[0], 12, 4000))
        pred = torch.argmax(pred)
        if pred != ytrue:
            print('Prediction: ', pred, 'Real: ', ytrue)

    y_pred = []
    y_true = []

    # iterate over test data
    for inputs, _, _, labels in val_loader:
        output = model(inputs.view(inputs.shape[0], 12, 4000)) # Feed Network

        output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
        y_pred.extend(output) # Save Prediction
        
        labels = labels.data.cpu().numpy()
        y_true.extend(labels) # Save Truth


    print('Accuracy: ', accuracy_score(y_true, y_pred))
    print('Precision: ', precision_score(y_true, y_pred))
    print('Recall: ', recall_score(y_true, y_pred))
    print('F1 Score: ', f1_score(y_true, y_pred))

    classes = ('Normal', 'Abnormal')
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix), index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.show()