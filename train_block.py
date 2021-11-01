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

from modules.mpdatagen import MPGraphDataGenerator
import modules.block as models
sns.set_context('poster')

if __name__=="__main__":

    os.environ['CUDA_VISIBLE_DEVICES']='1'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    args = parser.parse_args()

    config = yaml.load(open(args.config))

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
    already_preprocessed = data_config['already_preprocessed']

    concat_input = model_config['concat_input']

    epochs = train_config['epochs']
    learning_rate = train_config['learning_rate']
    alpha = train_config['alpha']
    save_dir = train_config['save_dir'] + '/Block_'+time.strftime("%Y%m%d")+'_concat'+str(concat_input)
    os.makedirs(save_dir, exist_ok=True)
    yaml.dump(config, open(save_dir + '/config.yaml', 'w'))

    # print('Running training for {} with concant_input: {}\n'.format(particle_type, concat_input))

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
        train_output_dir = output_dir + '/train/'
        val_output_dir = output_dir + '/val/'

        if already_preprocessed:
            train_files = np.sort(glob.glob(train_output_dir+'*.p'))[:num_train_files]
            val_files = np.sort(glob.glob(val_output_dir+'*.p'))[:num_val_files]

            pi0_train_files = train_files
            pi0_val_files = val_files
            pion_train_files = None
            pion_val_files = None


            train_output_dir = None
            val_output_dir = None

    data_gen_train = MPGraphDataGenerator(pi0_file_list=pi0_train_files,
                                        pion_file_list=pion_train_files,
                                        cellGeo_file=data_dir+'graph_examples/cell_geo.root',
                                        batch_size=batch_size,
                                        shuffle=shuffle,
                                        num_procs=num_procs,
                                        preprocess=preprocess,
                                        output_dir=train_output_dir)

    data_gen_val = MPGraphDataGenerator(pi0_file_list=pi0_val_files,
                                        pion_file_list=pion_val_files,
                                        cellGeo_file=data_dir+'graph_examples/cell_geo.root',
                                        batch_size=batch_size,
                                        shuffle=shuffle,
                                        num_procs=num_procs,
                                        preprocess=preprocess,
                                        output_dir=val_output_dir)

    if preprocess and not already_preprocessed:
        exit()

    # Optimizer.
    #optimizer = snt.optimizers.Adam(learning_rate)
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    model = models.BlockModel(global_output_size=2, model_config=model_config)

    training_loss_epoch = []
    val_loss_epoch = []
    
    checkpoint = tf.train.Checkpoint(module=model)
    checkpoint_prefix = os.path.join(save_dir, 'latest_model')
    latest = tf.train.latest_checkpoint(save_dir)
    if latest is not None:
        checkpoint.restore(latest)
    else:
        checkpoint.save(checkpoint_prefix)

    def convert_to_tuple(graphs):
        nodes = []
        edges = []
        globals = []
        senders = []
        receivers = []
        n_node = []
        n_edge = []
        offset = 0

        for graph in graphs:
            nodes.append(graph['nodes'])
            edges.append(graph['edges'])
            globals.append([graph['globals']])
            senders.append(graph['senders'] + offset)
            receivers.append(graph['receivers'] + offset)
            n_node.append(graph['nodes'].shape[:1])
            n_edge.append(graph['edges'].shape[:1])

            offset += len(graph['nodes'])

        nodes = tf.convert_to_tensor(np.concatenate(nodes))
        edges = tf.convert_to_tensor(np.concatenate(edges))
        globals = tf.convert_to_tensor(np.concatenate(globals))
        senders = tf.convert_to_tensor(np.concatenate(senders))
        receivers = tf.convert_to_tensor(np.concatenate(receivers))
        n_node = tf.convert_to_tensor(np.concatenate(n_node))
        n_edge = tf.convert_to_tensor(np.concatenate(n_edge))

        graph = GraphsTuple(
                nodes=nodes,
                edges=edges,
                globals=globals,
                senders=senders,
                receivers=receivers,
                n_node=n_node,
                n_edge=n_edge
            )

        return graph
       
    def get_batch(data_iter):
        for graphs, targets in data_iter:
            graphs = convert_to_tuple(graphs)
            targets = tf.convert_to_tensor(targets)
            
            yield graphs, targets

    samp_graph, samp_target = next(get_batch(data_gen_train.generator()))
    data_gen_train.kill_procs()
    graph_spec = utils_tf.specs_from_graphs_tuple(samp_graph, True, True, True)
    
    mae_loss = tf.keras.losses.MeanAbsoluteError()
    bce_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    def loss_fn(targets, predictions):
        return alpha * mae_loss(targets[:,:1], predictions[:,:1]) + (1 - alpha) * bce_loss(targets[:,1:], predictions[:,1:])

    @tf.function(input_signature=[graph_spec, tf.TensorSpec(shape=[None,2], dtype=tf.float32)])
    def train_step(graphs, targets):
        with tf.GradientTape() as tape:
            predictions = model(graphs).globals
            loss = loss_fn(targets, predictions)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return loss

    @tf.function(input_signature=[graph_spec, tf.TensorSpec(shape=[None,2], dtype=tf.float32)])
    def val_step(graphs, targets):
        predictions = model(graphs).globals
        loss = loss_fn(targets, predictions)

        return loss, predictions

    curr_loss = 1e5
    for e in range(epochs):

        print('\n\nStarting epoch: {}'.format(e))

        training_loss = []
        val_loss = []

        # Train
        print('Training...')
        i = 1
        for graph_data_tr, targets_tr in get_batch(data_gen_train.generator()):#train_iter):
            start = time.time()
            #if i==1:
            losses_tr = train_step(graph_data_tr, targets_tr)
            end = time.time()

            training_loss.append(losses_tr.numpy())

            print('Iter: {:03d}, Tr_loss_curr: {:.4f}, Tr_loss_mean: {:.4f}'. \
                  format(i, training_loss[-1], np.mean(training_loss)), end='\t')
            print('Took {:.3f} secs'.format(end-start))
            
            i += 1 
    
        training_loss_epoch.append(training_loss)

        # validate
        print('\nValidation...')
        i = 1
        all_targets = []
        all_outputs = []
        all_etas = []
        for graph_data_val, targets_val in get_batch(data_gen_val.generator()):#val_iter):
            start = time.time()
            losses_val, output_vals = val_step(graph_data_val, targets_val)
            end = time.time()

            targets_val = targets_val.numpy()
            output_vals = output_vals.numpy()

            targets_val[:,0] = 10**targets_val[:,0]
            output_vals[:,0] = 10**output_vals[:,0]
            targets_val[:,1] = 1 / (1 + np.exp(targets_val[:,1]))
            output_vals[:,1] = 1 / (1 + np.exp(output_vals[:,1]))

            val_loss.append(losses_val.numpy())
            all_targets.append(targets_val)
            all_outputs.append(output_vals)

            print('Iter: {:03d}, Val_loss_curr: {:.4f}, Val_loss_mean: {:.4f}'. \
                  format(i, val_loss[-1], np.mean(val_loss)), end='\t')
            print('Took {:.3f} secs'.format(end-start))
            
            i += 1 

        all_targets = np.concatenate(all_targets)
        all_outputs = np.concatenate(all_outputs)
        
        val_loss_epoch.append(val_loss)
    
        np.savez(save_dir+'/losses', training=training_loss_epoch, validation=val_loss_epoch)
        checkpoint.save(checkpoint_prefix)
    
        if np.mean(val_loss)<curr_loss:
            print('\nLoss decreased from {:.4f} to {:.4f}'.format(curr_loss, np.mean(val_loss)))
            print('Checkpointing and saving predictions to:\n{}'.format(save_dir))
            curr_loss = np.mean(val_loss)
            np.savez(save_dir+'/predictions', 
                    targets=all_targets, 
                    outputs=all_outputs)
            checkpoint.save(checkpoint_prefix)
