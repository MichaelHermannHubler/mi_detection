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
class UpscaleCNN(nn.Module):
    def __init__(self):
        super(UpscaleCNN, self).__init__()

        self.upscaler = nn.Sequential(         
            nn.Conv1d(in_channels=6, out_channels=12, kernel_size=1, stride=1, padding=0),                
            nn.ReLU()
        )

        # initialize the same architecture as the existing CNN model
        self.conv1 = utls.ConvolutionBlock(in_channels=12, out_channels=32)
        self.conv2 = utls.ConvolutionBlock(in_channels=32, out_channels=32)
        self.conv3 = utls.ConvolutionBlock(in_channels=32, out_channels=32)
        self.conv4 = utls.ConvolutionBlock(in_channels=32, out_channels=32)
        self.conv5 = utls.ConvolutionBlock(in_channels=32, out_channels=32)
        self.conv6 = utls.ConvolutionBlock(in_channels=32, out_channels=32)
        self.conv7 = utls.ConvolutionBlock(in_channels=32, out_channels=32)
        self.conv8 = utls.ConvolutionBlock(in_channels=32, out_channels=32) 

        self.lin1 = nn.Sequential(      
            nn.Dropout(p=0.2),
            nn.Linear(288, 512),
            nn.ReLU(),
        )        
        self.lin2 = nn.Sequential(      
            nn.Dropout(p=0.2),
            nn.Linear(512, 512),
            nn.ReLU(),
        )   
        self.lin3 = nn.Sequential(      
            nn.Dropout(p=0.2),
            nn.Linear(512, 192),
            nn.ReLU(),
        )

        self.norm = utls.OutputBlock(in_channels=192, out_channels=2)

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

        x = x.view(x.size(0), -1)

        x = self.lin1(x)
        x = self.lin2(x)
        x = self.lin3(x)
        
        return {
            'NORM': self.norm(x)
        }

def train(model, train_loader, optimizer, loss_fun, device, epoch):
    model.train()
    
    for i, (signal, _, _, targets) in enumerate(train_loader):        
        bs = signal.shape[0]
            
        x = signal.view(bs, 6, 5000)
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
            
        x = signal.view(bs, 6, 5000)
        x = x.to(device)
        
        # TODO

    model.eval(mode=False)

    return loss.item()

def criterion(loss_func, outputs, labels, device):
  losses = 0
  for i, key in enumerate(outputs):
    losses += loss_func(outputs[key], labels[key].to(device))
  return losses


def main():
    num_epochs = 100

    model = UpscaleCNN()

    # load the weights from the file
    state_dict = torch.load("G:\\Projects\\MA\\" + 'specbase_model1.chpt')    

    utls.loadConvolutionBlockLayer(state_dict, 'conv1', model.conv1)
    utls.loadConvolutionBlockLayer(state_dict, 'conv2', model.conv2)
    utls.loadConvolutionBlockLayer(state_dict, 'conv3', model.conv3)
    utls.loadConvolutionBlockLayer(state_dict, 'conv4', model.conv4)
    utls.loadConvolutionBlockLayer(state_dict, 'conv5', model.conv5)
    utls.loadConvolutionBlockLayer(state_dict, 'conv6', model.conv6)
    utls.loadConvolutionBlockLayer(state_dict, 'conv7', model.conv7)
    utls.loadConvolutionBlockLayer(state_dict, 'conv8', model.conv8)
    loadLayer(state_dict, 'lin1.1.', model.lin1[1])
    loadLayer(state_dict, 'lin2.1.', model.lin2[1])
    loadLayer(state_dict, 'lin3.1.', model.lin3[1])
    loadLayer(state_dict, 'norm.out.1.', model.norm.out[1])

    # freeze layers
    for name, param in model.named_parameters():
        if 'upscaler' not in name:
            param.requiresGrad = False

    optimizer = optim.Adam(params=model.parameters(), lr=0.000005)
    loss_fun = nn.CrossEntropyLoss()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    tr_loss = []
    tr_acc = []
    ev_loss = [0]
    ev_acc = [0]
    for epoch in range(num_epochs):
        loss = train(model, train_loader, optimizer, loss_fun, device, epoch)
        tr_loss.append(loss)
                
        # calculate accuracy
        model.eval()
        N = 200
        x, _, _, label = dataset[:N] 
        x = x.view(N, 6, 5000) if model.is_conv else  x.view(N,-1) 

        x = x.to(device)
        out = model(x)
        #out = torch.where(out < 0.5, torch.tensor(0).to(device), torch.tensor(1).to(device))

        dict_of_arrays = {}
        for key in label[0]:
            dict_of_arrays[key] = torch.tensor([d[key] for d in label], dtype=torch.float32).to(device)

        acc_ = (out['NORM'].argmax(-1) == dict_of_arrays['NORM']).float().sum()/len(label)
        acc_ = acc_.to('cpu')
        tr_acc.append(acc_)        
        
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
    layer.weight = torch.nn.Parameter(state_dict[layerName + 'weight'])
    layer.bias = torch.nn.Parameter(state_dict[layerName + 'bias'])


if __name__=="__main__":
    path = "G:\\Projects\\MA"
    model_version = 1

    dataset = PTBXLDataset(labels = ['NORM'], leads=range(6))
    train_loader, test_loader, val_loader = dataset.get_Loaders()

    recalculate = False

    if os.path.exists(os.path.join(path, f'upscalebase_model{model_version}.chpt')) and not recalculate:
        model = UpscaleCNN()
        model.load_state_dict(torch.load(os.path.join(path, f'upscalebase_model{model_version}.chpt')))
    else:
        model = main()
        torch.save(model.state_dict(), os.path.join(path, f'upscalebase_model{model_version}.chpt'))
        
        model = UpscaleCNN()
        model.load_state_dict(torch.load(os.path.join(path, f'upscalebase_model{model_version}.chpt')))

    model.eval()

    def data_test(model, signal, ytrue):
        pred = model(signal.view(signal.shape[0], 6, 5000))

        for i, key in enumerate(pred):
            accs.append(
                pred[key].argmax(-1) == ytrue[key]
            )
        

    y_pred = []
    y_true = []

    accs = []
    i = 0
    for inputs, _, _, labels in val_loader:
        data_test(model, inputs, labels)



    print(torch.tensor(accs, dtype=torch.float32).sum()/len(val_loader))

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