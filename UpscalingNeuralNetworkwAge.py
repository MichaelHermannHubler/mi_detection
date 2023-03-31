from ds_PTBXL import *
import Utils as utls

import torch.nn as nn
import seaborn as sn
import pandas as pd
import torch.optim as optim
from adabelief_pytorch import AdaBelief
import matplotlib.pyplot as plt
import os
import numpy as np

from torch.nn import BCEWithLogitsLoss
from tqdm import tqdm
from copy import deepcopy

from sklearn.metrics import confusion_matrix, accuracy_score

import torch.nn as nn
import torch
import pickle

from SpecialistNeuralNetworkwAge import CNN, FinetunedCNN

# create the new model
class UpscalingCNN(nn.Module):
    def __init__(self, specializationModel):
        super(UpscalingCNN, self).__init__()
        self.upscaler = nn.Sequential(         
            nn.Conv1d(in_channels=6, out_channels=12, kernel_size=50, stride=1, padding=25),                
            nn.LeakyReLU(),
            nn.Dropout1d(p=0.1),
            nn.BatchNorm1d(12),
        )

        self.specializationModel = specializationModel

        self.is_conv = True

    def forward(self, x, sex, age):
        x = self.upscaler(x)
        x = self.specializationModel(x, sex, age)

        return x


def train(model, train_loader, optimizer, loss_fun, device):
    model.train()
    run_loss = 0

    for i, (signal, sex, age, targets) in enumerate(tqdm(train_loader, desc='Train')):
        signal = signal.to(device)
        sex = sex.to(device)
        age = age.to(device)
        #targets = targets.values().to(device)
        labels = []
        for item in range(targets['NORM'].shape[0]):
            labels.append([
                targets['NORM'][item],
                targets['IMI'][item],
                targets['ASMI'][item],
                targets['ILMI'][item],
                targets['AMI'][item],
                targets['ALMI'][item],
                targets['INJAS'][item],
                targets['LMI'][item],
                targets['INJAL'][item],
                targets['IPLMI'][item],
                targets['IPMI'][item],
                targets['INJIN'][item],
                targets['INJLA'][item],
                targets['PMI'][item],
                targets['INJIL'][item],
            ])

        labels = torch.tensor(labels, dtype=torch.float32)
        labels = labels.to(device)

        # get batch size
        bs = signal.shape[0]

        # fully connected model: we need to flatten the signals
        x = signal.view(bs,-1) if not model.is_conv else signal.view(bs, 6, 4000)
        sex = sex.view(bs,-1) if not model.is_conv else sex.view(bs, 1)
        age = age.view(bs,-1) if not model.is_conv else age.view(bs, 1)

        # signal to device
        x = x.to(device)

        # zero grads
        optimizer.zero_grad()

        # forward pass
        out = model(x, sex, age)

        # calc loss and gradients

        loss = loss_fun(out, labels).mean()
        loss.backward()
        run_loss += loss.item()

        # update
        optimizer.step()

    return run_loss / len(train_loader)

def test(model, test_loader, device):
    model.eval()

    accs = []
    for i, (signal, sex, age, targets) in enumerate(tqdm(test_loader, desc='Test')):
        signal = signal.to(device)
        sex = sex.to(device)
        age = age.to(device)
        labels = []
        for item in range(targets['NORM'].shape[0]):
            labels.append([
                targets['NORM'][item],
                targets['IMI'][item],
                targets['ASMI'][item],
                targets['ILMI'][item],
                targets['AMI'][item],
                targets['ALMI'][item],
                targets['INJAS'][item],
                targets['LMI'][item],
                targets['INJAL'][item],
                targets['IPLMI'][item],
                targets['IPMI'][item],
                targets['INJIN'][item],
                targets['INJLA'][item],
                targets['PMI'][item],
                targets['INJIL'][item],
            ])

        labels = torch.tensor(labels, dtype=torch.float32)

        bs = signal.shape[0]

        x = signal.view(bs,-1) if not model.is_conv else signal.view(bs, 6, 4000)
        sex = sex.view(bs,-1) if not model.is_conv else sex.view(bs, 1)
        age = age.view(bs,-1) if not model.is_conv else age.view(bs, 1)
        x = x.to(device)

        out = model(x, sex, age)
        out = torch.sigmoid(out)
        out = torch.round(out)
        out = out.cpu().detach().numpy()

        acc_ = accuracy_score(labels, out)
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
    spezialisationModel.load_state_dict(torch.load("G:\\Projects\\MA\\models\\" + 'spec_modelfinal_withAge.chpt'))

    model = UpscalingCNN(spezialisationModel)

    return model

def main():
    num_epochs = 150
    num_folds = 5
    depth = '10.f'
    lr_adam = 5e-3

    best_acc = 0
    epoch_since_best = 0
    best_state = []

    for fold in range(num_folds):
        print(f'Fold [{fold + 1}/{num_folds}]:')
        model = loadModel()
        
        optimizer = optim.AdamW(params=model.parameters(), lr=lr_adam, eps=1e-7)
        lr = lr_adam

        loss_fun = nn.BCEWithLogitsLoss()

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
            tr_acc.append(test(model, trainacc_loader, device))
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
        plt.savefig(f'G:\Projects\MA\images\\UNN\{depth}_Layers_withAge_{fold + 1}_Loss_lr{lr}.png')
        plt.clf()

        plt.plot(tr_acc, label='train accuracy')
        plt.plot(te_acc, label='test accuracy')
        plt.legend()
        plt.savefig(f'G:\Projects\MA\images\\UNN\{depth}_Layers_withAge_{fold + 1}_Acc_lr{lr}.png')
        plt.clf()

        with open(f'G:\Projects\MA\\variables\\UNN\{depth}_Layers_withAge_{fold + 1}_train_loss.pkl', 'wb') as f:
            pickle.dump(tr_loss, f)

        with open(f'G:\Projects\MA\\variables\\UNN\{depth}_Layers_withAge_{fold + 1}_train_acc.pkl', 'wb') as f:
            pickle.dump(tr_acc, f)

        with open(f'G:\Projects\MA\\variables\\UNN\{depth}_Layers_withAge_{fold + 1}_test_acc.pkl', 'wb') as f:
            pickle.dump(te_acc, f)

    print(f'Best test acc = {best_acc:.5f}')
    model.load_state_dict(best_state)

    return model

def loadLayer(state_dict, layerName, layer):
    layer.weight = state_dict[layerName + 'weight']
    layer.bias = state_dict[layerName + 'bias']

if __name__=="__main__":
    train_dataset = PTBXLDataset(folds=range(6,8), leads=range(6))
    test_dataset = PTBXLDataset(folds=[9], leads=range(6))
    val_dataset = PTBXLDataset(folds=[10], leads=range(6))

    train_loader = train_dataset.get_full_loader(use_sampler=True)
    trainacc_loader = train_dataset.get_full_loader()
    test_loader = test_dataset.get_full_loader()
    val_loader = val_dataset.get_full_loader()

    path = "G:\\Projects\\MA\\models"
    model_version = 'final'

    recalculate = False

    if os.path.exists(os.path.join(path, f'upscale_model{model_version}_withAge.chpt')) and not recalculate:
        model = UpscalingCNN(FinetunedCNN(CNN()))
        model.load_state_dict(torch.load(os.path.join(path, f'upscale_model{model_version}_withAge.chpt')))
    else:
        model = main()
        torch.save(model.state_dict(), os.path.join(path, f'upscale_model{model_version}_withAge.chpt'))

        model = UpscalingCNN(FinetunedCNN(CNN()))
        model.load_state_dict(torch.load(os.path.join(path, f'upscale_model{model_version}_withAge.chpt')))

    model.eval()

    y_true = []
    y_pred = []

    def data_test(model, signal, ytrue, sex, age):
        signal = signal.view(signal.shape[0], 6, 4000)
        sex = sex.view(signal.shape[0], 1)
        age = age.view(signal.shape[0], 1)


        pred = model(signal, sex, age)
        pred = torch.sigmoid(pred)
        pred = torch.round(pred)
        pred = pred.cpu().detach().numpy()

        for i in range(len(pred)):
            accs['NORM'].append(pred[i][0] == ytrue['NORM'][i])
            accs['IMI'].append(pred[i][1] == ytrue['IMI'][i])
            accs['ASMI'].append(pred[i][2] == ytrue['ASMI'][i])
            accs['ILMI'].append(pred[i][3] == ytrue['ILMI'][i])
            accs['AMI'].append(pred[i][4] == ytrue['AMI'][i])
            accs['ALMI'].append(pred[i][5] == ytrue['ALMI'][i])
            accs['INJAS'].append(pred[i][6] == ytrue['INJAS'][i])
            accs['LMI'].append(pred[i][7] == ytrue['LMI'][i])
            accs['INJAL'].append(pred[i][8] == ytrue['INJAL'][i])
            accs['IPLMI'].append(pred[i][9] == ytrue['IPLMI'][i])
            accs['IPMI'].append(pred[i][10] == ytrue['IPMI'][i])
            accs['INJIN'].append(pred[i][11] == ytrue['INJIN'][i])
            accs['INJLA'].append(pred[i][12] == ytrue['INJLA'][i])
            accs['PMI'].append(pred[i][13] == ytrue['PMI'][i])
            accs['INJIL'].append(pred[i][14] == ytrue['INJIL'][i])

            count_pred['NORM'] += pred[i][0]
            count_pred['IMI']+= pred[i][1]
            count_pred['ASMI']+= pred[i][2]
            count_pred['ILMI']+= pred[i][3]
            count_pred['AMI']+= pred[i][4]
            count_pred['ALMI']+= pred[i][5]
            count_pred['INJAS']+= pred[i][6]
            count_pred['LMI']+= pred[i][7]
            count_pred['INJAL']+= pred[i][8]
            count_pred['IPLMI']+= pred[i][9]
            count_pred['IPMI']+= pred[i][10]
            count_pred['INJIN']+= pred[i][11]
            count_pred['INJLA']+= pred[i][12]
            count_pred['PMI']+= pred[i][13]
            count_pred['INJIL']+= pred[i][14]            

            count_true['NORM'] += ytrue['NORM'][i]
            count_true['IMI']+= ytrue['IMI'][i]
            count_true['ASMI']+= ytrue['ASMI'][i]
            count_true['ILMI']+= ytrue['ILMI'][i]
            count_true['AMI']+= ytrue['AMI'][i]
            count_true['ALMI']+= ytrue['ALMI'][i]
            count_true['INJAS']+= ytrue['INJAS'][i]
            count_true['LMI']+= ytrue['LMI'][i]
            count_true['INJAL']+= ytrue['INJAL'][i]
            count_true['IPLMI']+= ytrue['IPLMI'][i]
            count_true['IPMI']+= ytrue['IPMI'][i]
            count_true['INJIN']+= ytrue['INJIN'][i]
            count_true['INJLA']+= ytrue['INJLA'][i]
            count_true['PMI']+= ytrue['PMI'][i]
            count_true['INJIL']+= ytrue['INJIL'][i]

            y_true.append([
                targets['NORM'][i],
                targets['IMI'][i],
                targets['ASMI'][i],
                targets['ILMI'][i],
                targets['AMI'][i],
                targets['ALMI'][i],
                targets['INJAS'][i],
                targets['LMI'][i],
                targets['INJAL'][i],
                targets['IPLMI'][i],
                targets['IPMI'][i],
                targets['INJIN'][i],
                targets['INJLA'][i],
                targets['PMI'][i],
                targets['INJIL'][i],
            ])

            y_pred.append([
                pred[i][0],
                pred[i][1],
                pred[i][2],
                pred[i][3],
                pred[i][4],
                pred[i][5],
                pred[i][6],
                pred[i][7],
                pred[i][8],
                pred[i][9],
                pred[i][10],
                pred[i][11],
                pred[i][12],
                pred[i][13],
                pred[i][14]   
            ])

    count_pred = {
        'NORM': 0,
        'IMI': 0,
        'ASMI': 0,
        'ILMI': 0,
        'AMI': 0,
        'ALMI': 0,
        'INJAS': 0,
        'LMI': 0,
        'INJAL': 0,
        'IPLMI': 0,
        'IPMI': 0,
        'INJIN': 0,
        'INJLA': 0,
        'PMI': 0,
        'INJIL': 0,
    }

    count_true = {
        'NORM': 0,
        'IMI': 0,
        'ASMI': 0,
        'ILMI': 0,
        'AMI': 0,
        'ALMI': 0,
        'INJAS': 0,
        'LMI': 0,
        'INJAL': 0,
        'IPLMI': 0,
        'IPMI': 0,
        'INJIN': 0,
        'INJLA': 0,
        'PMI': 0,
        'INJIL': 0,
    }

    accs = {
        'NORM': [],
        'IMI': [],
        'ASMI': [],
        'ILMI': [],
        'AMI': [],
        'ALMI': [],
        'INJAS': [],
        'LMI': [],
        'INJAL': [],
        'IPLMI': [],
        'IPMI': [],
        'INJIN': [],
        'INJLA': [],
        'PMI': [],
        'INJIL': [],
    }
    i = 0
    
    for inputs, sex, age, labels in val_loader:
        data_test(model, inputs, labels, sex, age)

    for i, key in enumerate(accs):
        #print(accs[key])
        accs[key] = torch.stack(accs[key]).float().sum()/len(accs[key])
        print(f'{key}: {accs[key]:.3f}% ({count_pred[key]}/{count_true[key]})')
