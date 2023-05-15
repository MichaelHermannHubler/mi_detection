import Utils as utls
import ecg_plot
import os
import wfdb

dataset = utls.loadPTBXL()
pmi = dataset[dataset.PMI == 1]
record = wfdb.rdrecord(os.path.join("G:\\Projects\\MA\\data\\PTB-XL\\data", pmi.iloc[0]['filename_hr']))
ann = wfdb.rdann(os.path.join("G:\\Projects\\MA\\data\\PTB-XL\\data", pmi.iloc[0]['filename_hr']), 'dat')


wfdb.plot_items(signal=record.p_signal,
                    ann_samp=[ann.sample, ann.sample],
                    title='MIT-BIH Record 100', time_units='seconds',
                    figsize=(10,4), ecg_grids='all')