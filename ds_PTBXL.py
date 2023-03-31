import torch
import math
from Utils import *
from torch.utils.data import Dataset, DataLoader, Sampler
from sklearn.model_selection import train_test_split
import wfdb
import random
from functools import partial
from collections import defaultdict

class PTBXLDataset(Dataset):
    def __init__(self, folds = [], labels = [], leads = [], train_mode = False):
        super().__init__()
        self.path = "G:\\Projects\\MA\\data\\PTB-XL\\data"
        self.folds = folds if len(folds) > 0 else range(1,10)
        self.labels = labels if len(labels) > 0 else ['NORM', 'IMI', 'ASMI', 'ILMI', 'AMI', 'ALMI', 'INJAS', 'LMI', 'INJAL', 'IPLMI', 'IPMI', 'INJIN', 'INJLA', 'PMI', 'INJIL']
        self.leads = leads if len(leads) > 0 else range(12)
        self.train_mode = train_mode
        
        self.data = loadPTBXL()
        self.data = self.data[self.data['strat_fold'].isin(self.folds)]
        self.data.reset_index(inplace=True)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if self.train_mode:
            rand_val = np.random.randint(-250,+250)
        else:
            rand_val = 0

        # if self.train_mode and np.random.randint(0,2) == 1:
        #     multiplicator = -1
        # else:
        multiplicator = 1

        start = 500 + rand_val
        end = 4500 + rand_val

        if isinstance(idx, int):
            signals = wfdb.rdsamp(os.path.join(self.path, self.data['filename_hr'][idx]))

            signals = signals[0][start:end,self.leads]
            signals = signals * multiplicator
            # extract the signals and labels from the mat file
            sex = self.data['sex'][idx]
            age = self.data['age'][idx]
            label = {
                'NORM': None,
                'IMI': None,
                'ASMI': None,
                'ILMI': None,
                'AMI': None,
                'ALMI': None,
                'INJAS': None,
                'LMI': None,
                'INJAL': None,
                'IPLMI': None,
                'IPMI': None,
                'INJIN': None,
                'INJLA': None,
                'PMI': None,
                'INJIL': None,
                'MI':None
            }

            for key, value in label.items():
                if key in self.labels and key != 'MI':
                    label[key] = self.data[key][idx]
            
            label['MI'] = int( \
                self.data['IMI'][idx] + \
                self.data['ASMI'][idx] + \
                self.data['ILMI'][idx] + \
                self.data['AMI'][idx] + \
                self.data['ALMI'][idx] + \
                self.data['INJAS'][idx] + \
                self.data['LMI'][idx] + \
                self.data['INJAL'][idx] + \
                self.data['IPLMI'][idx] + \
                self.data['IPMI'][idx] + \
                self.data['INJIN'][idx] + \
                self.data['INJLA'][idx] + \
                self.data['PMI'][idx] + \
                self.data['INJIL'][idx] > 0
            )
            
            label = {k: v for k, v in label.items() if v is not None}

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
            signals = wfdb.rdsamp(os.path.join(self.path, self.data['filename_hr'][idx]))
            
            signals = signals[0][start:end,self.leads]
            signals = signals * multiplicator

            # extract the signals and labels from the mat file
            sex = self.data['sex'][idx]
            age = self.data['age'][idx]
            label = {
                'NORM': None,
                'IMI': None,
                'ASMI': None,
                'ILMI': None,
                'AMI': None,
                'ALMI': None,
                'INJAS': None,
                'LMI': None,
                'INJAL': None,
                'IPLMI': None,
                'IPMI': None,
                'INJIN': None,
                'INJLA': None,
                'PMI': None,
                'INJIL': None,
                'MI':None
            }

            for key, value in label.items():
                if key in self.labels and key != 'MI':
                    label[key] = self.data[key][idx]

            label['MI'] = int( \
                self.data['IMI'][idx] + \
                self.data['ASMI'][idx] + \
                self.data['ILMI'][idx] + \
                self.data['AMI'][idx] + \
                self.data['ALMI'][idx] + \
                self.data['INJAS'][idx] + \
                self.data['LMI'][idx] + \
                self.data['INJAL'][idx] + \
                self.data['IPLMI'][idx] + \
                self.data['IPMI'][idx] + \
                self.data['INJIN'][idx] + \
                self.data['INJLA'][idx] + \
                self.data['PMI'][idx] + \
                self.data['INJIL'][idx] > 0
            )
            
            label = {k: v for k, v in label.items() if v is not None}

            if math.isnan(age):
                age = 999

            signals = torch.tensor(signals, dtype=torch.float32)
            sex = torch.tensor(sex, dtype=torch.int64)
            age = torch.tensor(age, dtype=torch.int64)
            #label = torch.tensor(label, dtype=torch.float32)
            # return the signals and labels
            return signals, sex, age, label

    def get_loaders(self):
        train_data, test_data = train_test_split(self, test_size=0.30, random_state=42)
        test_data, val_data = train_test_split(test_data, test_size=0.50, random_state=42)

        return DataLoader(dataset=train_data, batch_size=32, shuffle=True), \
            DataLoader(dataset=test_data, batch_size=32, shuffle=True), \
            DataLoader(dataset=val_data, batch_size=1, shuffle=True)
    
    def get_full_loader(self, batch_size=32, use_sampler=False):
        # Strange performance increase - doing train test split before increases performance by up to ~30 times
        train_data, _ = train_test_split(self, test_size=0.00001, random_state=42)
        if use_sampler:
            bs = BalancedSampler(train_data, 100)
            return DataLoader(dataset=train_data, batch_size=batch_size, sampler=bs)
        else:
            return DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)

def display_first_10(dataset):
    for i in range(10):
        signals, sex, age, label = dataset[i]
        print("Signals: ", signals)
        print("Sex: ", sex)
        print("Age: ", age)
        print("Labels: ", label)

class BalancedSampler(Sampler):
    def __init__(self, dataset, num_samples=200):
        self.dataset = dataset

        label_counts = defaultdict(int)
        label_indices = defaultdict(list)
        for i in range(len(self.dataset)):
            _, _, _, label = self.dataset[i]
            for key in label.keys():
                label_counts[key] += label[key]
                if label[key] == 1:
                    label_indices[key].append(i)

        for key in label_counts.keys():
            if label_counts[key] < num_samples:
                label_indices[key] = random.choices(label_indices[key], k=num_samples)

        self.indices = []
        for key in label_counts.keys():
            self.indices.extend(label_indices[key])
        random.shuffle(self.indices)

        self.num_samples = len(self.indices)

    def __len__(self):
        return self.num_samples

    def __iter__(self):
        return iter(self.indices)

if __name__=="__main__":
    ds = PTBXLDataset(folds=[1], labels=['NORM', 'MI'])
    
    train_loader = ds.get_full_loader(use_sampler=True)
    label_counts = defaultdict(int)

    for i, (signal, _, _, targets) in enumerate(train_loader):
        print(targets)

