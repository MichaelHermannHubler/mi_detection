from copy import deepcopy
from ds_PTBXL import PTBXLDataset
from collections import defaultdict
import matplotlib.pyplot as plt
import pandas as pd
import json
from Utils import *
from scipy.signal import resample
import scipy.io
import seaborn as sn
import pickle

rootDirPath = 'G:\\Projects\\MA\\data\\'

def visualize_PTBXL_mi_class_distribution():
    full_dataset = PTBXLDataset(folds=range(1,8))
    full_loader = full_dataset.get_full_loader()

    specialist_dataset = PTBXLDataset(folds=range(1,5))
    specialist_loader = specialist_dataset.get_full_loader(use_sampler=True)

    upsampling_dataset = PTBXLDataset(folds=range(6,8))
    upsampling_loader = upsampling_dataset.get_full_loader(use_sampler=True)

    def demographics(loader):
        label_counts  = defaultdict(int)
        for i, (signal, sex, age, targets) in enumerate(loader):
            for i in range(len(targets['NORM'])):
                label_counts['NORM'] += targets['NORM'][i]
                label_counts['IMI']+= targets['IMI'][i]
                label_counts['ASMI']+= targets['ASMI'][i]
                label_counts['ILMI']+= targets['ILMI'][i]
                label_counts['AMI']+= targets['AMI'][i]
                label_counts['ALMI']+= targets['ALMI'][i]
                label_counts['INJAS']+= targets['INJAS'][i]
                label_counts['LMI']+= targets['LMI'][i]
                label_counts['INJAL']+= targets['INJAL'][i]
                label_counts['IPLMI']+= targets['IPLMI'][i]
                label_counts['IPMI']+= targets['IPMI'][i]
                label_counts['INJIN']+= targets['INJIN'][i]
                label_counts['INJLA']+= targets['INJLA'][i]
                label_counts['PMI']+= targets['PMI'][i]
                label_counts['INJIL']+= targets['INJIL'][i]
        
                label_counts['NORM_IMI_ASMI'] += targets['NORM'][i] | targets['IMI'][i] | targets['ASMI'][i]
                label_counts['Rest'] += targets['ILMI'][i] | targets['AMI'][i] | targets['ALMI'][i] | \
                                targets['INJAS'][i] | targets['LMI'][i] | targets['INJAL'][i] | \
                                targets['IPLMI'][i] | targets['IPMI'][i] | targets['INJIN'][i] | \
                                targets['INJLA'][i] | targets['PMI'][i] | targets['INJIL'][i]
                label_counts['Count'] += 1

        print(label_counts)
        return label_counts

    def merge_dict(dictA, dictB):
        """(dict, dict) => dict
    Merge two dicts, if they contain the same keys, it sums their values.
    Return the merged dict.
    Example:
        dictA = {'any key':1, 'point':{'x':2, 'y':3}, 'something':'aaaa'}
        dictB = {'any key':1, 'point':{'x':2, 'y':3, 'z':0, 'even more nested':{'w':99}}, 'extra':8}
        merge_dict(dictA, dictB)
        {'any key': 2,
         'point': {'x': 4, 'y': 6, 'z': 0, 'even more nested': {'w': 99}},
         'something': 'aaaa',
         'extra': 8}
    """
        r = {}

        common_k = [k for k in dictA if k in dictB]

        for k, v in dictA.items():
        # add unique k of dictA
            if k not in common_k:
                r[k] = v

            else:
            # add inner keys if they are not containing other dicts
                if type(v) is not dict:
                    if k in dictB:
                        r[k] = v + dictB[k]
                else:
                # recursively merge the inner dicts
                    r[k] = merge_dict(dictA[k], dictB[k])

    # add unique k of dictB
        for k, v in dictB.items():
            if k not in common_k:
                r[k] = v

        return r

    full_label_count = demographics(full_loader)
    specialist_label_count = demographics(specialist_loader)
    upsamling_label_count = demographics(upsampling_loader)

    combined_label_count = merge_dict(specialist_label_count, upsamling_label_count)

    print(full_label_count)
    print(specialist_label_count)
    print(upsamling_label_count)
    print(combined_label_count)

    def plotBarplot(label_count):
        label_count_cp = deepcopy(label_count)

        del label_count_cp['NORM_IMI_ASMI']
        del label_count_cp['Rest']
        del label_count_cp['Count']

        _, ax = plt.subplots()

        plt.bar(range(len(label_count_cp)), list(label_count_cp.values()), align='center')
        plt.xticks(range(len(label_count_cp)), list(label_count_cp.keys()))

        ax.set_xticklabels(ax.xaxis.get_majorticklabels(), rotation=45)
        plt.subplots_adjust(bottom=0.25)

        plt.show()

    plotBarplot(full_label_count)
    plotBarplot(specialist_label_count)
    plotBarplot(upsamling_label_count)
    plotBarplot(combined_label_count)

def visualize_PTBXL_mi_class_distribution():
    def loadPTBXLInfo() -> pd.DataFrame:
        info = pd.read_csv(rootDirPath +  'ptb-xl/data/scp_statements.csv')
        # Cut Without subclasses, as we need them for classification
        info = info[info['diagnostic_subclass'].notna()]
        return info
        
    ptb_info = loadPTBXLInfo()

    def PTBXLReplaceScpCode(scp: str) -> pd.DataFrame:
        res = json.loads(scp.replace("'", '"'))
        df = pd.DataFrame()

        for key in ptb_info['Unnamed: 0']:
            try:
                df[key] = [1 if res[key] > 0.5 else 0]
            except KeyError:
                df[key] = [0]

        return df

    path = rootDirPath + 'ptb-xl/data/ptbxl_database.csv'
    ptbxl_meta = pd.read_csv(path)
    interim = ptbxl_meta.apply(lambda row: PTBXLReplaceScpCode(row['scp_codes']), axis=1)
    interim = pd.concat(interim.values.tolist()).reset_index()
    ptbxl_meta = pd.concat([ptbxl_meta, interim], axis=1)

    ptbxl_meta['class'] = 'STTC'

    ptbxl_meta.loc[ptbxl_meta['LAFB'] + ptbxl_meta['IRBBB'] + ptbxl_meta['1AVB'] + ptbxl_meta['IVCD'] + ptbxl_meta['CRBBB'] +
                   ptbxl_meta['CLBBB'] + ptbxl_meta['LPFB'] + ptbxl_meta['WPW'] + ptbxl_meta['2AVB'] + ptbxl_meta['ILBBB'] +
                   ptbxl_meta['3AVB'] > 0, 'class'] = 'CD'
    
    ptbxl_meta.loc[ptbxl_meta['LVH'] + ptbxl_meta['LAO/LAE'] + ptbxl_meta['RVH'] + ptbxl_meta['RAO/RAE'] + ptbxl_meta['SEHYP'] > 0, 'class'] = 'HYP'

    ptbxl_meta.loc[ptbxl_meta['IMI'] + ptbxl_meta['ASMI'] + ptbxl_meta['ILMI'] + ptbxl_meta['AMI'] + ptbxl_meta['ALMI'] + ptbxl_meta['INJAS'] + 
                   ptbxl_meta['LMI'] + ptbxl_meta['INJAL'] + ptbxl_meta['IPLMI'] + ptbxl_meta['IPMI'] + ptbxl_meta['INJIN'] + ptbxl_meta['INJLA'] + 
                   ptbxl_meta['PMI'] + ptbxl_meta['INJIL'] > 0, 'class'] = 'MI'
    
    ptbxl_meta.loc[ptbxl_meta['NORM'] > 0, 'class'] = 'NORM'
    
    class_counts = ptbxl_meta['class'].value_counts().sort_index()
    plt.bar(class_counts.index, class_counts.values)
    plt.xlabel('Class')
    plt.ylabel('Frequency')
    plt.title('Distribution of Classes')

    plt.show()
    #print(ptbxl_meta.head().T)
    #print(ptbxl_meta.groupby('class').count())

def visualize_Combined_dataset():
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

    def getSignal(path : str, source : str) -> np.ndarray:
        if source == 'C15':
            exam, file = path.split('@')
            signal = getTracingFromH5(exam, file)
            # transform from (4096, 12) to (12, 4096)
            signal = signal.T
            # Upsample, as the data is only taken in 400Hz instead of 500Hz
            signal = upsample_array(signal)[:,:]

        if source == "CP":
            mat = scipy.io.loadmat(path)
            signal =  mat['ECG'][0][0][2][:,:]
        
        if source in ('CPE', 'CS', 'GE', 'NI'):
            mat = scipy.io.loadmat(path)
            signal = mat['val'][:,:]

        # max_val = np.max(np.abs(signal))
        # signal = signal / np.linalg.norm(signal)

        # signal = normalize(signal)

        return signal

    def getSignalLength(row: pd.Series) -> Tuple[str, int]:
        return getSignal(row['fileName'], row['source']).size/12

    dataChap = loadChapmanShaoxing()
    dataChap['source'] = 'CS'

    dataCode = loadCode15()
    dataCode['source'] = 'C15'
    #dataCode = dataCode.sample(n=int(len(dataCode) / 10), random_state=42)

    dataCPSC = loadCPSC2018()
    dataCPSC['source'] = 'CP'

    # remove "unclean Data"
    dataCPSC['Signal_Length'] = dataCPSC.apply(getSignalLength, axis = 1)
    print('CPSC', len(dataCPSC[dataCPSC['Signal_Length'] < 5000]))
    print('CPSC', len(dataCPSC[dataCPSC['Signal_Length'] >= 5000]))
    dataCPSC = dataCPSC[dataCPSC['Signal_Length'] >= 5000]
    dataCPSC.reset_index(drop=True, inplace=True)
    dataCPSC.drop(columns=['Signal_Length'], inplace=True)

    dataCPSCExtra = loadCPSC2018Extra()
    dataCPSCExtra['source'] = 'CPE'

    # remove "unclean Data"
    dataCPSCExtra['Signal_Length'] = dataCPSCExtra.apply(getSignalLength, axis = 1)
    print('CPSC E', len(dataCPSCExtra[dataCPSCExtra['Signal_Length'] < 5000]))
    print('CPSC E', len(dataCPSCExtra[dataCPSCExtra['Signal_Length'] >= 5000]))
    dataCPSCExtra = dataCPSCExtra[dataCPSCExtra['Signal_Length'] >= 5000]
    dataCPSCExtra.reset_index(drop=True, inplace=True)
    dataCPSCExtra.drop(columns=['Signal_Length'], inplace=True)

    dataGeorgia = loadGeorgia()
    dataGeorgia['source'] = 'GE'

    # remove "unclean Data"
    dataGeorgia['Signal_Length'] = dataGeorgia.apply(getSignalLength, axis = 1)
    print('Georgia', len(dataGeorgia[dataGeorgia['Signal_Length'] < 5000]))
    print('Georgia', len(dataGeorgia[dataGeorgia['Signal_Length'] >= 5000]))
    dataGeorgia = dataGeorgia[dataGeorgia['Signal_Length'] >= 5000]
    dataGeorgia.reset_index(drop=True, inplace=True)
    dataGeorgia.drop(columns=['Signal_Length'], inplace=True)

    dataNingbo = loadNingbo()
    dataNingbo['source'] = 'NI'
    #dataNingbo = dataNingbo.sample(n=int(len(dataNingbo) / 2), random_state=42)
    
    df = pd.concat([dataChap, dataCode, dataCPSC, dataCPSCExtra, dataGeorgia, dataNingbo])

    print(df.groupby('source').count())
    df.dropna(inplace=True)
    print(df.groupby('source').count())
    
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
    df = df_balanced

    df.loc[df['diagnostics'] == 0, 'diagnostics'] = 'Normal'
    df.loc[df['diagnostics'] == 1, 'diagnostics'] = 'Abnormal'

    df.loc[df['sex'] == 0, 'sex'] = 'Male'
    df.loc[df['sex'] == 1, 'sex'] = 'Female'

    counts = df.groupby(['source', 'diagnostics']).size().reset_index(name='count')

    # pivot the data to make it easier to plot
    pivoted = counts.pivot(index='diagnostics', columns='source', values='count')

    # create the bar plot
    ax = pivoted.plot(kind='bar', stacked=True)

    # add labels and title
    ax.set_xlabel('Diagnosis')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of diagnostics by source')
    ax.set_xticklabels(labels=ax.get_xticklabels(),rotation=0)

    # show the plot
    plt.show()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

    # create distribution plot of age column in first subplot
    df['age'].plot(kind='hist', ax=ax1)
    ax1.set_xlabel('Age')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Age')

    # create bar plot of sex column, segmented by source, in second subplot
    df.groupby(['sex', 'source']).size().unstack().plot(kind='bar', stacked=True, ax=ax2)
    ax2.set_xlabel('Sex')
    ax2.set_ylabel('Count')
    ax2.set_title('Sex Segmented by Source')
    ax2.xaxis.get_label().set_rotation(0)
    ax2.set_xticklabels(labels=ax2.get_xticklabels(),rotation=0)

    # set overall title for the plot
    fig.suptitle('Distribution of Age and Sex Segmented by Source')

    # display the plot
    plt.show()

    print(df_balanced.head())

    dataChap, dataCode, dataCPSC, dataCPSCExtra, dataGeorgia, dataNingbo

    print(dataCode.groupby('sex').count())
    print(dataCode.describe())
    
    print(dataCPSC.groupby('sex').count())
    print(dataCPSC.describe())
    
    print(dataCPSCExtra.groupby('sex').count())
    print(dataCPSCExtra.describe())
    
    print(dataGeorgia.groupby('sex').count())
    print(dataGeorgia.describe())
    
    print(dataChap.groupby('sex').count())
    print(dataChap.describe())
    
    print(dataNingbo.groupby('sex').count())
    print(dataNingbo.describe())

def visualize_confmat_bin():
    classNo = ['Normal', 'Abnormal']
    classMI = ['MI', 'non-MI']
    name = ['confmat_Upscaling_Base', 'confmat_Specialist_Base', 'confmat_generalization_with_age', 'confmat_generalization_without_age']
    classList = [classMI, classMI, classNo, classNo]

    cf_matrices = [
        [[0.27,0.098],[0.077,0.56]],
        [[0.57,0.063],[0.1,0.26]],
        [[0.49,0.016],[0.064,0.43]],
        [[0.49,0.019],[0.062,0.43]],
    ]

    for i in range(len(name)):
        print(name[i])
        cf_matrix = cf_matrices[i]
        df_cm = pd.DataFrame(cf_matrix/np.sum(cf_matrix), index = [j for j in classList[i]],
                            columns = [j for j in classList[i]])
        plt.figure(figsize = (12,7))
        sn.heatmap(df_cm, annot=True,  annot_kws={"size": 10})
        plt.show()
        
def visualize_confmat_mult():
    name = ['confmat_spec', 'confmat_spec_metadata', 'confmat_upscaling1a', 'confmat_upscaling1b', 'confmat_upscaling1c', 'confmat_upscaling2']
    classList = ['NORM', 'IMI', 'ASMI', 'ILMI', 'AMI', 'ALMI', 'INJAS', 'LMI', 'INJAL', 'IPLMI', 'PMI', 'INJIN', 'INJLA', 'PMI', 'INJIL']

    cf_matrices = [
        [
            [362,191,73,890],
            [1172,76,194,74],
            [1240,40,79,157],
            [1468,0,48,0],
            [1481,0,35,0],
            [1487,0,29,0],
            [1492,2,21,1],
            [1496,0,20,0],
            [1500,2,13,1],
            [1511,0,5,0],
            [1513,0,3,0],
            [1514,0,2,0],
            [1514,0,2,0],
            [1514,0,2,0],
            [1514,0,2,0]
        ],
        [
            [350,203,63,900],
            [1104,144,140,128],
            [1235,45,81,155],
            [1468,0,48,0],
            [1481,0,35,0],
            [1487,0,29,0],
            [1494,0,22,0],
            [1496,0,20,0],
            [1502,0,14,0],
            [1511,0,5,0],
            [1513,0,3,0],
            [1514,0,2,0],
            [1514,0,2,0],
            [1514,0,2,0],
            [1514,0,2,0]
        ],
        [
            [383,170,71,892],
            [1153,95,142,126],
            [1237,43,156,80],
            [1468,0,48,0],
            [1481,0,35,0],
            [1487,0,29,0],
            [1494,0,22,0],
            [1495,1,20,0],
            [1502,0,14,0],
            [1510,1,5,0],
            [1513,0,3,0],
            [1514,0,2,0],
            [1514,0,2,0],
            [1514,0,2,0],
            [1514,0,2,0]
        ],
        [
            [371,182,227,736],
            [1090,158,189,79],
            [1176,104,180,56],
            [1468,0,48,0],
            [1481,0,35,0],
            [1487,0,29,0],
            [1489,5,22,0],
            [1496,0,20,0],
            [1502,0,14,0],
            [1511,0,5,0],
            [1513,0,3,0],
            [1514,0,2,0],
            [1514,0,2,0],
            [1514,0,2,0],
            [1513,1,2,0]
        ],
        [
            [431,122,136,827],
            [1107,141,127,141],
            [1201,79,166,70],
            [1468,0,48,0],
            [1481,0,35,0],
            [1487,0,29,0],
            [1494,0,22,0],
            [1496,0,20,0],
            [1502,0,14,0],
            [1511,0,5,0],
            [1513,0,3,0],
            [1514,0,2,0],
            [1514,0,2,0],
            [1514,0,2,0],
            [1514,0,2,0]
        ],
        [
            [366,187,88,875],
            [1155,93,157,111],
            [1178,102,137,99],
            [1458,10,46,2],
            [1481,0,35,0],
            [1487,0,29,0],
            [1493,1,22,0],
            [1492,4,20,0],
            [1502,0,14,0],
            [1511,0,5,0],
            [1513,0,3,0],
            [1514,0,2,0],
            [1514,0,2,0],
            [1514,0,2,0],
            [1514,0,2,0]
        ]
    ]

    for i in range(len(name)):
        print(name[i])
        if np.sum(cf_matrices[i]) != 22740:
            print('ERROR')

        f, axes = plt.subplots(3, 5, figsize=(25, 15))
        axes = axes.ravel()
        
        for j in range(len(cf_matrices[i])):
            cf_matrix = [[cf_matrices[i][j][0], cf_matrices[i][j][1]],
                         [cf_matrices[i][j][2], cf_matrices[i][j][3]]]
            
            labels =  ['Others', classList[j]]
        
            disp = sn.heatmap(cf_matrix, ax=axes[j], annot=True, fmt='.4g', 
                              xticklabels=labels, yticklabels=labels,  annot_kws={"size": 18}, cbar=False)
            
        plt.subplots_adjust(wspace=0.10, hspace=0.20)
        plt.show()

def visualize_optimizers():
    ada = sgd = adam = []

    with open(f'G:\Projects\MA\\variables\GNN\AdaGradOptim_Layers_1_train_acc.pkl', 'rb') as f:
        ada = pickle.load(f)
    with open(f'G:\Projects\MA\\variables\GNN\SGDOptim_Layers_1_train_acc.pkl', 'rb') as f:
        sgd = pickle.load(f)
    with open(f'G:\Projects\MA\\variables\GNN\AdamOptim_Layers_1_train_acc.pkl', 'rb') as f:
         adam = pickle.load(f)

    plt.plot(ada, label="AdaGrad")
    plt.plot(sgd, label="SGD")
    plt.plot(adam, label="Adam")
    plt.legend(loc="lower right")
    plt.xlim(0,35)
    plt.show()

#visualize_PTBXL_mi_class_distribution()
#visualize_PTBXL_mi_class_distribution()
#visualize_Combined_dataset()
#visualize_confmat_bin()
#visualize_confmat_mult()
#visualize_optimizers()


rootDirPath = 'G:\\Projects\\MA\\data\\'
path = rootDirPath + 'cpsc2018-extra/data/WFDB_CPSC2018_2/'

fileNames = []
Ages = []
Sexes = []
Diagnostics = []

for child in Path(rootDirPath + 'ningbo/data/WFDB_Ningbo/').glob('*.mat'):     
    fileNames.append(child)
    data = wfdb.rdsamp(os.path.splitext(child)[0])
    Ages.append(data[1]['comments'][0].split(' ')[1])
    Sexes.append(data[1]['comments'][1].split(' ')[1])
    
meta = pd.DataFrame(data={'fileName':fileNames, 'age':Ages, 'sex':Sexes})
print(meta.groupby('sex').count())
print(meta.describe())
#print('END')