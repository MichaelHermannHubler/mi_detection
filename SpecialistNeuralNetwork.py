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

import torch.nn as nn
import torch

# create the new model
class FinetunedCNN(nn.Module):
    def __init__(self):
        super(FinetunedCNN, self).__init__()

        # initialize the same architecture as the existing CNN model
        self.conv1 = utls.ConvolutionBlock(in_channels=12, out_channels=32)
        self.conv2 = utls.ConvolutionBlock(in_channels=32, out_channels=32)
        self.conv3 = utls.ConvolutionBlock(in_channels=32, out_channels=32)
        self.conv4 = utls.ConvolutionBlock(in_channels=32, out_channels=32)
        self.conv5 = utls.ConvolutionBlock(in_channels=32, out_channels=32)
        self.conv6 = utls.ConvolutionBlock(in_channels=32, out_channels=32)
        self.conv7 = utls.ConvolutionBlock(in_channels=32, out_channels=32)
        self.conv8 = utls.ConvolutionBlock(in_channels=32, out_channels=32) 

        self.norm = utls.OutputBlock(in_channels=192, out_channels=2)
        self.imi = utls.OutputBlock(in_channels=192, out_channels=2)
        self.asmi = utls.OutputBlock(in_channels=192, out_channels=2)
        self.ilmi = utls.OutputBlock(in_channels=192, out_channels=2)
        self.ami = utls.OutputBlock(in_channels=192, out_channels=2)
        self.almi = utls.OutputBlock(in_channels=192, out_channels=2)
        self.injas = utls.OutputBlock(in_channels=192, out_channels=2)
        self.lmi = utls.OutputBlock(in_channels=192, out_channels=2)
        self.injal = utls.OutputBlock(in_channels=192, out_channels=2)
        self.iplmi = utls.OutputBlock(in_channels=192, out_channels=2)
        self.ipmi = utls.OutputBlock(in_channels=192, out_channels=2)
        self.injin = utls.OutputBlock(in_channels=192, out_channels=2)
        self.injla = utls.OutputBlock(in_channels=192, out_channels=2)
        self.pmi = utls.OutputBlock(in_channels=192, out_channels=2)
        self.injil = utls.OutputBlock(in_channels=192, out_channels=2)

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

        x = x.view(x.size(0), -1)
        
        return {
            'NORM': self.norm(x),
            'IMI': self.imi(x),
            'ASMI': self.asmi(x),
            'ILMI': self.ilmi(x),
            'AMI': self.ami(x),
            'ALMI': self.almi(x),
            'INJAS': self.injas(x),
            'LMI': self.lmi(x),
            'INJAL': self.injal(x),
            'IPLMI': self.iplmi(x),
            'IPMI': self.ipmi(x),
            'INJIN': self.injin(x),
            'INJLA': self.injla(x),
            'PMI': self.pmi(x),
            'INJIL': self.injil(x)
        }

def train(model, train_loader, optimizer, loss_fun, device, epoch):
    model.train()
    
    for i, (signal, _, _, targets) in enumerate(train_loader):        
        bs = signal.shape[0]
            
        x = signal.view(bs, 12, 5000)
        x = x.to(device)
            
        optimizer.zero_grad()
        out = model(x)
        loss = criterion(loss_fun, out, targets, device).mean()
        loss.backward()
            
        optimizer.step()

    model.train(mode=False)

    return loss.item()

def test(model, train_loader, optimizer, loss_fun, device, epoch):
    model.eval()

    for i, (signal, _, _, target) in enumerate(train_loader):
        bs = signal.shape[0]
            
        x = signal.view(bs, 12, 5000)
        x = x.to(device)
        
        # TODO

    model.eval(mode=False)

    return loss.item()

def criterion(loss_func, outputs, labels, device):
  losses = 0
  for i, key in enumerate(outputs):
    print(outputs)
    print(labels)

    losses += loss_func(outputs[key], labels[key].to(device))
  return losses


def main():
    num_epochs = 100

    model = FinetunedCNN()

    # load the weights from the file
    state_dict = torch.load("G:\\Projects\\MA\\" + 'model4.chpt')

    loadLayer(state_dict, 'conv1.0.', model.conv1)
    loadLayer(state_dict, 'conv2.0.', model.conv2)
    loadLayer(state_dict, 'conv3.0.', model.conv3)
    loadLayer(state_dict, 'conv4.0.', model.conv4)
    loadLayer(state_dict, 'conv5.0.', model.conv5)
    loadLayer(state_dict, 'conv6.0.', model.conv6)
    loadLayer(state_dict, 'conv7.0.', model.conv7)
    loadLayer(state_dict, 'conv8.0.', model.conv8)

    # freeze layers
    for name, param in model.named_parameters():
        if 'conv1' in name or 'conv2' in name or 'conv3' in name or 'conv4' in name or 'conv5' in name:
            param.requiresGrad = False

    print(model)

    optimizer = optim.Adam(params=model.parameters(), lr=0.0005)
    loss_fun = nn.CrossEntropyLoss()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    tr_loss = []
    tr_acc = []
    ev_loss = []
    ev_acc = []
    for epoch in range(num_epochs):
        loss = train(model, train_loader, optimizer, loss_fun, device, epoch)
        tr_loss.append(loss)
                
        # calculate accuracy
        model.eval()
        N = 2000
        x, _, _, label = dataset[:N] 
        x = x.view(N, 12, 5000) if model.is_conv else  x.view(N,-1) 

        x = x.to(device)
        label = label.to(device)
        out = model(x)
        #out = torch.where(out < 0.5, torch.tensor(0).to(device), torch.tensor(1).to(device))

        acc_ = utls.hamming_score(label, out)
        #acc_ = ((label & out).sum(axis=1) / (label | out).sum(axis=1)).mean()
        #acc_ = (out == label).float().sum()/len(label)
        acc_ = acc_.to('cpu')
        tr_acc.append(acc_)


        x, _, _, label = dataset[:N] 
        x = x.view(N, 12, 5000) if model.is_conv else  x.view(N,-1)
        model.eval()
        
        x = x.to(device)
        label = label.to(device)
        out = model(x)
        #out = torch.where(out < 0.5, torch.tensor(0).to(device), torch.tensor(1).to(device))

        acc_ = utls.hamming_score(label, out)
        #acc_ = ((label & out).sum(axis=1) / (label | out).sum(axis=1)).mean()
        #acc_ = (out == label).float().sum()/len(label)
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

def loadLayer(state_dict, layerName, layer):
    layer.weight = state_dict[layerName + 'weight']
    layer.bias = state_dict[layerName + 'bias']

dataset = PTBXLDataset()
train_loader, test_loader, val_loader = dataset.get_Loaders()
path = "G:\\Projects\\MA"
model_version = 1

recalculate = True

if os.path.exists(os.path.join(path, f'spec_model{model_version}.chpt')) and not recalculate:
    model = FinetunedCNN()
    model.load_state_dict(torch.load(os.path.join(path, f'spec_model{model_version}.chpt')))
else:
    model = main()
    torch.save(model.state_dict(), os.path.join(path, f'spec_model{model_version}.chpt'))
    
    model = FinetunedCNN()
    model.load_state_dict(torch.load(os.path.join(path, f'spec_model{model_version}.chpt')))

model.eval()


def data_test(model, signal, ytrue):
    pred = model(signal.view(signal.shape[0], 12, 5000))
    #pred = torch.argmax(pred)
    # if pred != ytrue:
    print('Prediction: ', pred, 'Real: ', ytrue)

y_pred = []
y_true = []


i = 0
for inputs, _, _, labels in val_loader:
    if labels[0][0] == torch.tensor(0) and i < 10:
        data_test(model, inputs, labels)
        i = i+1


# def multilabel_confusion_matrix(y_true, y_pred):
#     y_true = np.array(y_true)
#     y_pred = np.array(y_pred)
#     assert y_true.shape == y_pred.shape, "Input shapes do not match"
#     assert len(y_true.shape) == 2, "Input should be 2D arrays"

#     n_classes = y_true.shape[1]
#     conf_mat = np.zeros((n_classes, n_classes))

#     for i in range(n_classes):
#         for j in range(n_classes):
#             true_positives = np.sum((y_true[:, i] == 1) & (y_pred[:, j] == 1))
#             false_positives = np.sum((y_true[:, i] == 0) & (y_pred[:, j] == 1))
#             false_negatives = np.sum((y_true[:, i] == 1) & (y_pred[:, j] == 0))
#             conf_mat[i, j] = true_positives, false_positives, false_negatives
#     return conf_mat

# # iterate over test data
# for inputs, _, _, labels in val_loader:
#     output = model(inputs.view(inputs.shape[0], 12, 5000)) # Feed Network

#     output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
#     y_pred.extend(output) # Save Prediction
    
#     labels = labels.data.cpu().numpy()
#     y_true.extend(labels) # Save Truth


# conf_mat = multilabel_confusion_matrix(y_true, y_pred)

# df_cm = pd.DataFrame(conf_mat, range(23), range(23))
# sn.set(font_scale=1.4)
# sn.heatmap(df_cm, annot=True, annot_kws={"size": 16})
# plt.show()

# classes = (
#     'NORM','STTC','NST_','IMI','AMI','LVH','LAFB/LPFB','ISC_','IRBBB','_AVB','IVCD','ISCA','CRBBB','CLBBB',
#     'LAO/LAE','ISCI','LMI','RVH','RAO/RAE','WPW','ILBBB','SEHYP','PMI')
# cf_matrix = confusion_matrix(y_true, y_pred)
# df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix), index = [i for i in classes],
#                      columns = [i for i in classes])
# plt.figure(figsize = (12,7))
# sn.heatmap(df_cm, annot=True)
# plt.show()