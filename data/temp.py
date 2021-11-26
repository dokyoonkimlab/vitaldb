import pandas as pd
import os
from shutil import copyfile,copy



internal_dir = '../../ALL_OBS10MID10_tupdated/'

data = pd.read_csv('example_test_data.csv',sep=',')


for filename in data['filename']:
    copy(internal_dir + filename, 'test_intraop/')

data['ID'] = ['P' + str(i) for i in data.index]

for filename,ID in zip(data['filename'],data['ID']):
    os.rename('test_intraop/' + filename, 'test_intraop/' + ID + '.csv')
data['filename'] = [ID + '.csv' for ID in data['ID']]
d = data[config.preop_features + ['masstf','ID','filename']]
