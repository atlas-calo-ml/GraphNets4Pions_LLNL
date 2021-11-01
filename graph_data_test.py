import numpy as np
import os
import glob
import uproot as ur
import matplotlib.pyplot as plt
import time
import seaborn as sns
# import tensorflow as tf
from modules.graph_data import GraphDataGenerator
sns.set_context('poster')

data_dir = '/usr/workspace/hip/ML4Jets/regression_images/'
pion_files = np.sort(glob.glob(data_dir+'*graphs.v01*/*singlepion*/*root'))
data = ur.open(pion_files[0])
data_eventTree = data['EventTree']
data_cellGeo = ur.open(data_dir+'graph_examples/cell_geo.root')['CellGeo']


data_gen = GraphDataGenerator(file_list=pion_files[:2], 
                              cellGeo_file=data_dir+'graph_examples/cell_geo.root',
                              batch_size=32,
                              shuffle=False)

for g, t in data_gen.gen:
    pass

data_gen.restart()

for g, t in data_gen.gen:
    pass
