from typing import Tuple
from pathlib import Path

import wfdb
import pandas as pd
import scipy.io as sio
import json
import os
import torch
import torch.nn as nn

rootDirPath = 'G:\\Projects\\MA\\'

def loadCode15() -> pd.DataFrame:
    path = rootDirPath + 'Code-15/exams.csv'
    meta = pd.read_csv(path)
    meta['sex'] = 'Male'
    meta.loc[meta['is_male'] == False, 'sex'] = 'Female'
    meta = meta.drop(columns=['nn_predicted_age', '1dAVb', 'RBBB', 'LBBB', 'SB', 'ST', 'AF', 'patient_id', 'death', 'timey'])
    return meta

def loadCPSC2018() -> pd.DataFrame:
    if os.path.exists(os.path.join(rootDirPath, "cpsc2018_data.pkl")):
        cpsc2018_meta_cleaned = pd.read_pickle(os.path.join(rootDirPath, "cpsc2018_data.pkl"))
    else:
        path = rootDirPath + 'cpsc2018/Reference.csv'
        cpsc2018_meta = pd.read_csv(path)
        cpsc2018_meta_cleaned = cpsc2018_meta[['Recording', 'First_label']]
        
        # reduce only to Elements with a size of 5000
        cpsc2018_meta_cleaned['Signal_Length'] = cpsc2018_meta_cleaned.apply(getCPSC2018SignalLengthDataFromMatlab, axis = 1)
        cpsc2018_meta_cleaned = cpsc2018_meta_cleaned[cpsc2018_meta_cleaned['Signal_Length'] == 5000]
        cpsc2018_meta_cleaned = cpsc2018_meta_cleaned.reset_index(drop=True)

        cpsc2018_meta_cleaned['sex'] = cpsc2018_meta_cleaned.apply(getCPSC2018SexDataFromMatlab, axis = 1)
        cpsc2018_meta_cleaned['age'] = cpsc2018_meta_cleaned.apply(getCPSC2018AgeDataFromMatlab, axis = 1)

        # reduce outputs to normal (0), abnormal (1) as we dont need the detailed information
        cpsc2018_meta_cleaned.loc[cpsc2018_meta_cleaned['First_label'] == 1, 'First_label'] = 0
        cpsc2018_meta_cleaned.loc[cpsc2018_meta_cleaned['First_label'] > 1, 'First_label'] = 1

        cpsc2018_meta_cleaned.loc[cpsc2018_meta_cleaned['sex'] == 'Male', 'sex'] = 0
        cpsc2018_meta_cleaned.loc[cpsc2018_meta_cleaned['sex'] != 'Male', 'sex'] = 1

        cpsc2018_meta_cleaned['label'] = cpsc2018_meta_cleaned['First_label']
        cpsc2018_meta_cleaned.drop('First_label', axis=1, inplace=True)
        cpsc2018_meta_cleaned.drop('Signal_Length', axis=1, inplace=True)

        cpsc2018_meta_cleaned.to_pickle(os.path.join(rootDirPath, "cpsc2018_data.pkl"))

    return cpsc2018_meta_cleaned

def loadCPSC2018Extra() -> pd.DataFrame:
    path = rootDirPath + 'physionetchallenge/cpsc2018-extra/data/WFDB_CPSC2018_2/'

    fileNames = []
    Ages = []
    Sexes = []
    Diagnostics = []

    for child in Path(path).glob('*.mat'):     
        fileNames.append(child)
        data = wfdb.rdsamp(os.path.splitext(child)[0])
        Ages.append(data[1]['comments'][0].split(' ')[1])
        Sexes.append(data[1]['comments'][1].split(' ')[1])
        Diagnostics.append(convertSNOMEDCTCodeToBinaryLabel(data[1]['comments'][2].split(' ')[1].split(',')))

    meta = pd.DataFrame(data={'fileName':fileNames, 'age':Ages, 'sex':Sexes, 'diagnostics':Diagnostics})
    meta.age = meta.age.astype('float')

    return meta

def loadChapmanShaoxing() -> Tuple[pd.DataFrame, pd.DataFrame]:
    fileNames = []
    Ages = []
    Sexes = []
    Diagnostics = []

    for child in Path('../Datasets/physionetchallenge/chapman-shaoxing/data/').glob('*.mat'):     
        fileNames.append(child)
        data = wfdb.rdsamp(os.path.splitext(child)[0])
        Ages.append(data[1]['comments'][0].split(' ')[1])
        Sexes.append(data[1]['comments'][1].split(' ')[1])
        Diagnostics.append(convertSNOMEDCTCodeToBinaryLabel(data[1]['comments'][2].split(' ')[1].split(',')))

    meta = pd.DataFrame(data={'fileName':fileNames, 'age':Ages, 'sex':Sexes, 'diagnostics':Diagnostics})
    meta.age = meta.age.astype('float')

    return meta

def loadGeorgia() -> Tuple[pd.DataFrame, pd.DataFrame]:
    fileNames = []
    Ages = []
    Sexes = []
    Diagnostics = []

    for child in Path('../Datasets/physionetchallenge/georgiachallenge/data/WFDB_Ga/').glob('*.mat'):     
        fileNames.append(child)
        data = wfdb.rdsamp(os.path.splitext(child)[0])
        Ages.append(data[1]['comments'][0].split(' ')[1])
        Sexes.append(data[1]['comments'][1].split(' ')[1])
        Diagnostics.append(convertSNOMEDCTCodeToBinaryLabel(data[1]['comments'][2].split(' ')[1].split(',')))

    meta = pd.DataFrame(data={'fileName':fileNames, 'age':Ages, 'sex':Sexes, 'diagnostics':Diagnostics})
    meta.age = meta.age.astype('float')

    return meta

def loadNingbo() -> Tuple[pd.DataFrame, pd.DataFrame]:
    fileNames = []
    Ages = []
    Sexes = []
    Diagnostics = []

    for child in Path('../Datasets/physionetchallenge/ningbo/data/WFDB_Ningbo/').glob('*.mat'):     
        fileNames.append(child)
        data = wfdb.rdsamp(os.path.splitext(child)[0])
        Ages.append(data[1]['comments'][0].split(' ')[1])
        Sexes.append(data[1]['comments'][1].split(' ')[1])
        Diagnostics.append(convertSNOMEDCTCodeToBinaryLabel(data[1]['comments'][2].split(' ')[1].split(',')))

    meta = pd.DataFrame(data={'fileName':fileNames, 'age':Ages, 'sex':Sexes, 'diagnostics':Diagnostics})
    meta.age = meta.age.astype('float')

    return meta

def loadPTBXLInfo() -> pd.DataFrame:
    info = pd.read_csv(rootDirPath +  'ptb-xl/data/scp_statements.csv')
    # Cut Without subclasses, as we need them for classification
    info = info[info['diagnostic_subclass'].notna()]
    return info
    
ptb_info = loadPTBXLInfo()

def loadPTBXL() -> pd.DataFrame:    
    if os.path.exists(os.path.join(rootDirPath, "ptbxl_data.pkl")):
        ptbxl_meta = pd.read_pickle(os.path.join(rootDirPath, "ptbxl_data.pkl"))
    else:
        ptbxl_meta = pd.read_csv(rootDirPath +  'ptb-xl/data/ptbxl_database.csv')
        #gender 0 is male, 1 is female

        # Convert SCP Codes
        interim = ptbxl_meta.apply(lambda row: PTBXLReplaceScpCode(row['scp_codes']), axis=1)
        interim = pd.concat(interim.values.tolist()).reset_index()

        ptbxl_meta = pd.concat([ptbxl_meta, interim], axis=1)
        ptbxl_meta = ptbxl_meta[
            ptbxl_meta['NORM'] + 
            ptbxl_meta['IMI'] + 
            ptbxl_meta['ASMI'] + 
            ptbxl_meta['ILMI'] + 
            ptbxl_meta['AMI'] + 
            ptbxl_meta['ALMI'] + 
            ptbxl_meta['INJAS'] + 
            ptbxl_meta['LMI'] + 
            ptbxl_meta['INJAL'] + 
            ptbxl_meta['IPLMI'] + 
            ptbxl_meta['IPMI'] + 
            ptbxl_meta['INJIN'] + 
            ptbxl_meta['INJLA'] + 
            ptbxl_meta['PMI'] + 
            ptbxl_meta['INJIL'] > 0
        ]

        ptbxl_meta = ptbxl_meta[
            [
                'ecg_id', 'patient_id', 'age', 'sex', 'height', 'weight', 'scp_codes', 'strat_fold', 'filename_lr', 'filename_hr',
                'NORM', 'IMI', 'ASMI', 'ILMI', 'AMI', 'ALMI', 'INJAS', 'LMI', 'INJAL', 'IPLMI', 'IPMI', 'INJIN', 'INJLA', 'PMI', 'INJIL'
            ]]
        ptbxl_meta.reset_index(inplace=True)
        ptbxl_meta.to_pickle(os.path.join(rootDirPath, "ptbxl_data.pkl"))

    return ptbxl_meta

def PTBXLReplaceScpCode(scp: str) -> pd.DataFrame:
    res = json.loads(scp.replace("'", '"'))
    df = pd.DataFrame()

    for key in ptb_info['Unnamed: 0']:
        try:
            df[key] = [res[key]/100.0]
        except KeyError:
            df[key] = [0]

    return df

def convertAgeToAgeBin(age: int) -> int:
    if age < 30:
        return 0
    elif age < 40:
        return 1
    elif age < 50:
        return 2
    elif age < 60:
        return 3
    elif age < 70:
        return 4
    elif age < 80:
        return 5
    else:
        return 6

def getCPSC2018SexDataFromMatlab(row: pd.Series) -> Tuple[str, int]:
    file = row['Recording']
    mat_contents = sio.loadmat(os.path.join(rootDirPath, f'cpsc2018/data/{file}.mat'))

    return mat_contents['ECG'][0][0][0][0]

def getCPSC2018AgeDataFromMatlab(row: pd.Series) -> Tuple[str, int]:
    file = row['Recording']
    mat_contents = sio.loadmat(os.path.join(rootDirPath, f'cpsc2018/data/{file}.mat'))

    return mat_contents['ECG'][0][0][1][0][0]

def getCPSC2018SignalLengthDataFromMatlab(row: pd.Series) -> Tuple[str, int]:
    file = row['Recording']
    mat_contents = sio.loadmat(os.path.join(rootDirPath, f'cpsc2018/data/{file}.mat'))

    return mat_contents['ECG'][0][0][2].size/12

def convertSNOMEDCTCodeToBinaryLabel(code: int) -> int:
    switch = { # using https://browser.ihtsdotools.org/?perspective=full&conceptId1=404684003&edition=MAIN/2022-09-30&release=&languages=en
               # disorders and abnormalities = 1
               # pure findings and others = 0
        '427172004':1,
        '284470004':1,
        '426627000':1,
        '164909002':1,
        '82226007':1,
        '233917008':1,
        '164921003':1,
        '251120003':1,
        '195126007':1,
        '59931005':1,
        '251259000':1,
        '49578007':1,
        '10370003':0,
        '426783006':0,
        '270492004':1,
        '427393009':1,
        '47665007':1,
        '370365005':1,
        '426177001':1,
        '413444003':1,
        '713427006':1,
        '426995002':1,
        '164896001':1,
        '251164006':1,
        '426761007':1,
        '266249003':1,
        '426749004':1,
        '251170000':1,
        '63593006':1,
        '164867002':1,
        '81898007':1,
        '195042002':1,
        '164917005':1,
        '65778007':1,
        '59118001':1,
        '17338001':1,
        '698252002':1,
        '164930006':1,
        '164895002':1,
        '67741000119109':1,
        '55930002':1,
        '164934002':1,
        '164865005':1,
        '427084000':0,
        '27885002':1,
        '164931005':0,
        '164861001':1,
        '429622005':0,
        '446813000':1,
        '75532003':1,
        '428750005':1,
        '195060002':1,
        '54329005':1,
        '89792004':1,
        '111288001':1,
        '251180001':1,
        '77867006':0,
        '164937009':1,
        '111975006':1,
        '164889003':1,
        '11157007':1,
        '446358003':1,
        '29320008':1,
        '164890007':1,
        '74615001':1,
        '713422000':1,
        '713426002':1,
        '413844008':1,
        '426648003':1,
        '704997005':1,
        '164873001':1,
        '195080001':1,
        '251146004':0,
        '39732003':1,
        '55827005':1,
        '251199005':0,
        '164912004':1,
        '428417006':0,
        '251198002':0,
        '67751000119106':1,
        '164942001':0,
        '74390002':1,
        '164947007':1,
        '251173003':1,
        '54016002':1,
        '251166008':1,
        '17366009':1,
        '233897008':1,
        '445118002':1,
        '426434006':1,
        '425419005':1,
        '425623009':1,
        '253352002':1,
        '426664006':0,
        '195101003':1,
        '253339007':1,
        '251266004':0,
        '251268003':0,
        '445211001':1,
        '164884008':0,

        '365413008':0,
        '251223006':1,
        '425856008':0,
        '106068003':0,
        '61721007':1,
        '733534002':1,
        '6374002':1,
        '61277005':1,
        '426183003':1,
        '50799005':1,
        '251187003':1,
        '57054005':1,
        '13640000':1,
        '418818005':1,
        '233892002':1,
        '251205003':1,
        '5609005':1
    }

    for c in code:
        if switch[c] == 1:
            return 1

    return 0

class ConvolutionBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvolutionBlock, self).__init__()
        self.conv = nn.Sequential(         
            nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=2, padding=10),                
            nn.ReLU(),                      
            nn.MaxPool1d(kernel_size=2),    
            nn.BatchNorm1d(out_channels)
        )
    def forward(self, x):
        x = self.conv(x)
        return x

class OutputBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutputBlock, self).__init__()
        self.out = nn.Sequential(         
            nn.Dropout(p=0.2),
            nn.Linear(in_channels, out_channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.out(x)
        return x

def hamming_score(y_true, y_pred, normalize=True):
    y_true = y_true.view(-1)
    y_pred = y_pred.view(-1)
    score = torch.sum(y_true != y_pred).float()
    if normalize:
        score /= y_true.size(0)
    return score

if __name__=="__main__":
    print('Main')

    #data = loadCode15()
    data = loadPTBXL()
    print(data.head(500))


