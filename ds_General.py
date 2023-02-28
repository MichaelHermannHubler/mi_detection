import scipy.io
import torch
import math
from Utils import *
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.signal import resample


class GeneralDataset(Dataset):
    def __init__(self, include_sex = False, include_age = False):
        super().__init__()
        self.path = "G:\\Projects\\MA\\"
        self.include_sex = include_sex
        self.include_age = include_age

        self.data = self.initData()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if isinstance(idx, int):
            signals = self.getSignal(self.data['fileName'][idx], self.data['source'][idx])

            if self.include_sex:
                sex = self.data['sex'][idx]
            else:
                sex = 0
                
            if self.include_age:
                age = self.data['age'][idx]
            else:
                age = 0

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

            if self.include_sex:
                sex = self.data['sex'][idx]
            else:
                sex = 0
                
            if self.include_age:
                age = self.data['age'][idx]
            else:
                age = 0

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
        train_size = int(0.7 * len(self))
        test_size = int(0.15 * len(self))
        val_size = len(self) - train_size - test_size
        train_data, test_data, val_data = torch.utils.data.random_split(self, [train_size, test_size, val_size])


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

        if not self.include_sex:
            df.drop(['sex'], axis=1, inplace=True)
        else:
            df.drop(df.loc[df['sex'] == 'Unknown'].index, inplace=True)

        if not self.include_age:
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

        df_balanced.reset_index(inplace=True)
        
        return df_balanced
    
    def getSignal(self, path : str, source : str) -> np.ndarray:
        if source == 'C15':
            exam, file = path.split('@')
            signal = getTracingFromH5(exam, file)
            # transform from (4096, 12) to (12, 4096)
            signal = signal.T
            # Upsample, as the data is only taken in 400Hz instead of 500Hz
            signal = upsample_array(signal)[:,:5000]

        if source == "CP":
            mat = scipy.io.loadmat(path)
            signal =  mat['ECG'][0][0][2][:,:5000]
        
        if source in ('CPE', 'CS', 'GE', 'NI'):
            mat = scipy.io.loadmat(path)
            signal = mat['val'][:,:5000]

        max_val = np.max(np.abs(signal))
        signal = signal / max_val

        return signal
    
def upsample_array(arr):
    # Get the current and target sampling rates
    cur_sr = 400
    target_sr = 500
    
    # Calculate the resampling ratio
    ratio = target_sr / cur_sr
    
    # Get the number of samples in the current array
    n_samples = arr.shape[1]
    
    # Create an array of target sample indices
    target_indices = np.arange(n_samples * ratio) / ratio
    
    # Resample the array
    resampled_arr = resample(arr, target_indices.shape[0], axis=1)
    
    return resampled_arr

def display_first_10(dataset):
    for i in range(10):
        signals, sex, age, label = dataset[i]
        print("Signals: ", signals)
        print("Sex: ", sex)
        print("Age: ", age)
        print("Labels: ", label)


if __name__=="__main__":
    ds = GeneralDataset()    

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

    df.drop(['sex'], axis=1, inplace=True)
    df.drop(['age'], axis=1, inplace=True)
    df.dropna(inplace=True)

    for index, row in df.iterrows():
        signal = ds.getSignal(row['fileName'], row['source'])
        if signal.shape[0] != 12 or signal.shape[1] < 5000:
            print(row)
            print(ds.getSignal(row['fileName'], row['source']).T.shape)
            break