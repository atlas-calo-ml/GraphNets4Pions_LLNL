import numpy as np
import os
import sys
import glob
import uproot as ur
import matplotlib.pyplot as plt
import time
import seaborn as sns
import tensorflow as tf
from graph_nets import utils_np
from graph_nets import utils_tf
from graph_nets.graphs import GraphsTuple
import sonnet as snt
import argparse
import yaml
import logging
import tensorflow as tf

from modules.mpdatagen_nearest import MPGraphDataGeneratorMultiOut
import modules.multiOutBlock_wWeightedRegress as models
sns.set_context('poster')

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    parser.add_argument('--k', default=10)
    args = parser.parse_args()

    config = yaml.load(open(args.config))
    k = int(args.k)

    data_config = config['data']
    model_config = config['model']
    train_config = config['training']

    data_dir = data_config['data_dir']
    num_train_files = data_config['num_train_files']
    num_val_files = data_config['num_val_files']
    batch_size = data_config['batch_size']
    shuffle = data_config['shuffle']
    num_procs = data_config['num_procs']
    preprocess = data_config['preprocess']
    output_dir = data_config['output_dir']
    use_xyz = data_config['use_xyz']
    already_preprocessed = data_config['already_preprocessed']

    concat_input = model_config['concat_input']

    epochs = train_config['epochs']
    learning_rate = train_config['learning_rate']
    alpha = train_config['alpha']
    os.environ['CUDA_VISIBLE_DEVICES'] = str(train_config['gpu'])
    log_freq = train_config['log_freq']

    pi0_files = np.sort(glob.glob(data_dir+'*graphs.v01*/*pi0*/*root'))
    pion_files = np.sort(glob.glob(data_dir+'*graphs.v01*/*pion*/*root'))
    train_start = 10
    train_end = train_start + num_train_files
    val_end = train_end + num_val_files
    pi0_train_files = pi0_files[train_start:train_end]
    pi0_val_files = pi0_files[train_end:val_end]
    pion_train_files = pion_files[train_start:train_end]
    pion_val_files = pion_files[train_end:val_end]

    train_output_dir = None
    val_output_dir = None
            
    # Get Data
    if preprocess:
        if use_xyz:
            train_output_dir = output_dir + '/nearest/xyz/k_'+str(k)+'/train/'
            val_output_dir = output_dir + '/nearest/xyz/k_'+str(k)+'/val/'
        else:
            train_output_dir = output_dir + '/nearest/eta_phi_rPerp/k_'+str(k)+'/train/'
            val_output_dir = output_dir + '/nearest/eta_phi_rPerp/k_'+str(k)+'/val/'


    data_gen_train = MPGraphDataGeneratorMultiOut(pi0_file_list=pi0_train_files,
                                        pion_file_list=pion_train_files,
                                        cellGeo_file=data_dir+'graph_examples/cell_geo.root',
                                        batch_size=batch_size,
                                        k=k,
                                        shuffle=shuffle,
                                        num_procs=num_procs,
                                        preprocess=preprocess,
                                        output_dir=train_output_dir)

    data_gen_val = MPGraphDataGeneratorMultiOut(pi0_file_list=pi0_val_files,
                                        pion_file_list=pion_val_files,
                                        cellGeo_file=data_dir+'graph_examples/cell_geo.root',
                                        batch_size=batch_size,
                                        k=k,
                                        shuffle=shuffle,
                                        num_procs=num_procs,
                                        preprocess=preprocess,
                                        output_dir=val_output_dir)
    print('Done for k = {}!!'.format(k))
