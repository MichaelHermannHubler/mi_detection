import Utils as utls
import ecg_plot
import os
import wfdb

dataset = utls.loadPTBXL()
pmi = dataset[dataset.PMI == 1]
signal = wfdb.rdsamp(os.path.join("G:\\Projects\\MA\\data\\PTB-XL\\data", pmi.iloc[0]['filename_hr']))[0]

wfdb.plot_items(signal, figsize=(10,4), ecg_grids='all')