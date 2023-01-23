import scipy.io
import torch
import math
from Utils import *
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import ecg_plot
import wfdb
import numpy as np

class PTBXLDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.path = "G:\\Projects\\MA\\PTB-XL\\data"
        self.data = loadPTBXL()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            signals = wfdb.rdsamp(os.path.join(self.path, self.data['filename_hr'][idx]))[0][:,:5000]
            # extract the signals and labels from the mat file
            sex = self.data['sex'][idx]
            age = self.data['age'][idx]
            label = {
                'NORM': self.data['NORM'][idx],
                'IMI': self.data['IMI'][idx],
                'ASMI': self.data['ASMI'][idx],
                'ILMI': self.data['ILMI'][idx],
                'AMI': self.data['AMI'][idx],
                'ALMI': self.data['ALMI'][idx],
                'INJAS': self.data['INJAS'][idx],
                'LMI': self.data['LMI'][idx],
                'INJAL': self.data['INJAL'][idx],
                'IPLMI': self.data['IPLMI'][idx],
                'IPMI': self.data['IPMI'][idx],
                'INJIN': self.data['INJIN'][idx],
                'INJLA': self.data['INJLA'][idx],
                'PMI': self.data['PMI'][idx],
                'INJIL': self.data['INJIL'][idx]
            }

            if math.isnan(age):
                age = 999

            signals = torch.tensor(signals, dtype=torch.float32)
            sex = torch.tensor(sex, dtype=torch.int64)
            age = torch.tensor(age, dtype=torch.int64)
            #label = torch.tensor(label, dtype=torch.float32)
            # return the signals and labels
            return signals, sex, age, label

        elif isinstance(idx, slice):
            start = idx.start if idx.start != None else 0
            stop = idx.stop if idx.stop != None else len(self)
            step = idx.step if idx.step != None else 1
            
            return_signals = []
            return_sex = []
            return_age = []
            return_label = []

            for i in range(start, stop, step):
                signal, sex, age, label = self[i]
                return_signals.append(signal)
                return_sex.append(sex)
                return_age.append(age)
                return_label.append(label)

                
            return_signals = torch.stack(return_signals)
            return_sex = torch.stack(return_sex)
            return_age = torch.stack(return_age)
            #return_label = torch.stack(return_label)

            return return_signals, return_sex, return_age, return_label
        else:
            signals = wfdb.rdsamp(os.path.join(self.path, self.data['filename_hr'][idx]))[0][:,:5000]
            # extract the signals and labels from the mat file
            sex = self.data['sex'][idx]
            age = self.data['age'][idx]
            label = {
                'NORM': self.data['NORM'][idx],
                'IMI': self.data['IMI'][idx],
                'ASMI': self.data['ASMI'][idx],
                'ILMI': self.data['ILMI'][idx],
                'AMI': self.data['AMI'][idx],
                'ALMI': self.data['ALMI'][idx],
                'INJAS': self.data['INJAS'][idx],
                'LMI': self.data['LMI'][idx],
                'INJAL': self.data['INJAL'][idx],
                'IPLMI': self.data['IPLMI'][idx],
                'IPMI': self.data['IPMI'][idx],
                'INJIN': self.data['INJIN'][idx],
                'INJLA': self.data['INJLA'][idx],
                'PMI': self.data['PMI'][idx],
                'INJIL': self.data['INJIL'][idx]
            }

            if math.isnan(age):
                age = 999

            signals = torch.tensor(signals, dtype=torch.float32)
            sex = torch.tensor(sex, dtype=torch.int64)
            age = torch.tensor(age, dtype=torch.int64)
            #label = torch.tensor(label, dtype=torch.float32)
            # return the signals and labels
            return signals, sex, age, label

    def get_Loaders(self):
        train_data, test_data = train_test_split(self, test_size=0.30, random_state=42)
        test_data, val_data = train_test_split(test_data, test_size=0.50, random_state=42)

        return DataLoader(dataset=train_data, batch_size=32, shuffle=True), \
            DataLoader(dataset=test_data, batch_size=32, shuffle=True), \
            DataLoader(dataset=val_data, batch_size=1, shuffle=True)


def display_first_10(dataset):
    for i in range(10):
        signals, sex, age, label = dataset[i]
        print("Signals: ", signals)
        print("Sex: ", sex)
        print("Age: ", age)
        print("Labels: ", label)


if __name__=="__main__":
    print('Main')
    
    dataset = PTBXLDataset()
    signal, _, _, label = dataset[1]
    print(label)