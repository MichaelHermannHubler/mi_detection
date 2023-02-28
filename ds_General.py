import scipy.io
import torch
import math
from Utils import *
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import ecg_plot

class GeneralDataset(Dataset):
    def __init__(self, includeSex = False, includeAge = False):
        super().__init__()
        self.path = "G:\\Projects\\MA\\"
        self.includeSex = includeSex
        self.includeAge = includeAge

        self.data = self.initData()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            signals = self.getSignal(self.data['fileName'][idx], self.data['source'][idx])

            if self.includeSex:
                sex = self.data['sex'][idx]
            else:
                sex = []
                
            if self.includeAge:
                age = self.data['age'][idx]
            else:
                age = []

            label = self.data['diagnostics'][idx]
            # convert the signals and labels to tensors

            if math.isnan(age):
                age = 999

            signals = torch.tensor(signals, dtype=torch.float32)
            sex = torch.tensor(sex, dtype=torch.int64)
            age = torch.tensor(age, dtype=torch.int64)
            label = torch.tensor(label, dtype=torch.int64)
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
            return_label = torch.stack(return_label)

            return return_signals, return_sex, return_age, return_label
        else:
            signals = self.getSignal(self.data['fileName'][idx], self.data['source'][idx])

            if self.includeSex:
                sex = self.data['sex'][idx]
            else:
                sex = []
                
            if self.includeAge:
                age = self.data['age'][idx]
            else:
                age = []
                
            label = self.data['diagnostics'][idx]
            # convert the signals and labels to tensors

            if math.isnan(age):
                age = 999

            signals = torch.tensor(signals, dtype=torch.float32)
            sex = torch.tensor(sex, dtype=torch.int64)
            age = torch.tensor(age, dtype=torch.int64)
            label = torch.tensor(label, dtype=torch.int64)
            # return the signals and labels
            return signals, sex, age, label

    def get_Loaders(self):
        train_data, test_data = train_test_split(self, test_size=0.30, random_state=42)
        test_data, val_data = train_test_split(test_data, test_size=0.50, random_state=42)

        return DataLoader(dataset=train_data, batch_size=32, shuffle=True), \
            DataLoader(dataset=test_data, batch_size=32, shuffle=True), \
            DataLoader(dataset=val_data, batch_size=1, shuffle=True)
    
    def initData(self) -> pd.DataFrame:
        dataChap = loadChapmanShaoxing()
        dataChap['source'] = 'CS'

        dataCode = loadCode15()
        dataCode['source'] = 'C15'

        dataCPSC = loadCPSC2018()
        dataCPSC['source'] = 'CP'

        dataCPSCExtra = loadCPSC2018Extra()
        dataCPSCExtra['source'] = 'CPE'

        dataGeorgia = loadGeorgia()
        dataGeorgia['source'] = 'GE'

        dataNingbo = loadNingbo()
        dataNingbo['source'] = 'NI'

        df = pd.concat([dataChap, dataCode, dataCPSC, dataCPSCExtra, dataGeorgia, dataNingbo])

        if not self.includeSex:
            df.drop(['sex'], axis=1, inplace=True)
        else:
            df.drop(df.loc[df['sex'] == 'Unknown'].index, inplace=True)

        if not self.includeAge:
            df.drop(['age'], axis=1, inplace=True)

        df.dropna(inplace=True)
        
        # Separate the data into two dataframes based on the value of the diagnostics column
        df_0 = df[df['diagnostics'] == 0]
        df_1 = df[df['diagnostics'] == 1]

        # Calculate the minimum number of rows to select from each dataframe
        min_rows = min(len(df_0), len(df_1))

        # Sample an equal number of rows from each dataframe
        df_0_sampled = df_0.sample(n=min_rows, random_state=42)
        df_1_sampled = df_1.sample(n=min_rows, random_state=42)

        # Concatenate the two dataframes into a single, balanced dataframe
        df_balanced = pd.concat([df_0_sampled, df_1_sampled])
        return df_balanced
    
    def getSignal(self, path : str, source : str) -> np.ndarray:
        print(path)

        if source == 'C15':
            exam, file = path.split('@')
            signal = getTracingFromH5(exam, file)[:,:5000]

        if source == "CP":
            mat = scipy.io.loadmat(path)
            signal =  mat['ECG'][0][0][2][:,:5000]
        
        if source in ('CPE', 'CS', 'GE', 'NI'):
            mat = scipy.io.loadmat(path)
            signal = mat['val'][:,:5000]

        max_val = np.max(np.abs(signal))
        signal = signal / max_val

        return signal


def display_first_10(dataset):
    for i in range(10):
        signals, sex, age, label = dataset[i]
        print("Signals: ", signals)
        print("Sex: ", sex)
        print("Age: ", age)
        print("Labels: ", label)


if __name__=="__main__":
    ds = GeneralDataset()
    c15 = ds.data[ds.data['source'] == 'C15'].iloc[0]
    cp = ds.data[ds.data['source'] == 'CP'].iloc[0]
    cpe = ds.data[ds.data['source'] == 'CPE'].iloc[0]
    cs = ds.data[ds.data['source'] == 'CS'].iloc[0]
    ge = ds.data[ds.data['source'] == 'GE'].iloc[0]
    ni = ds.data[ds.data['source'] == 'NI'].iloc[0]
    
    print('C15', ds.getSignal(c15.fileName, c15.source))
    print('CP', ds.getSignal(cp.fileName, cp.source))
    print('CPE', ds.getSignal(cpe.fileName, cpe.source))
    print('CS', ds.getSignal(cs.fileName, cs.source))
    print('GE', ds.getSignal(ge.fileName, ge.source))
    print('NI', ds.getSignal(ni.fileName, ni.source))