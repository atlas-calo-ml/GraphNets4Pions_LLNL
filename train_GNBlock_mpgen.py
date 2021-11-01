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
import pickle
import argparse

from modules.mpdatagen import MPGraphDataGenerator
import modules.gnBlock as models
sns.set_context('poster')

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--concat', action='store_true')
    parser.add_argument('--particle_type')
    args = parser.parse_args()
    concat_input = args.concat
    particle_type = args.particle_type
    print('Running training for {} with concant_input: {}\n'.format(particle_type, concat_input))
            
    # Get Data
    data_dir = '/usr/workspace/hip/ML4Jets/regression_images/'
    pion_files = np.sort(glob.glob(data_dir+'*graphs.v01*/*'+particle_type+'*/*root'))

    epochs = 200
    batch_size = 1024

    data_gen_train = MPGraphDataGenerator(file_list=pion_files[10:20],
                                        cellGeo_file=data_dir+'graph_examples/cell_geo.root',
                                        batch_size=batch_size,
                                        preprocess=True,
                                        output_dir='preprocessed_data/'+particle_type+'/train/',
                                        shuffle=False)

    data_gen_val = MPGraphDataGenerator(file_list=pion_files[20:22],
                                      cellGeo_file=data_dir+'graph_examples/cell_geo.root',
                                      batch_size=batch_size,
                                      preprocess=True,
                                      output_dir='preprocessed_data/'+particle_type+'/val/',
                                      shuffle=False)

    # Optimizer.
    learning_rate = 1e-3
    #optimizer = snt.optimizers.Adam(learning_rate)
    optimizer = tf.keras.optimizers.Adam(learning_rate)

    model = models.GNBlockModel(global_output_size=1, concat_input=concat_input)

    training_loss_epoch = []
    val_loss_epoch = []

    save_dir = 'results/gnBlock_'+time.strftime("%Y%m%d")+'eve_concat'+str(concat_input)
    os.makedirs(save_dir, exist_ok=True)
    
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
    
    loss_fn = tf.keras.losses.MeanAbsoluteError()

    @tf.function(input_signature=[graph_spec, tf.TensorSpec(shape=[None,1], dtype=tf.float32)])
    def train_step(graphs, targets):
        with tf.GradientTape() as tape:
            predictions = 4*tf.nn.tanh(model(graphs).globals)
            loss = loss_fn(targets, predictions)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return loss

    @tf.function(input_signature=[graph_spec, tf.TensorSpec(shape=[None,1], dtype=tf.float32)])
    def val_step(graphs, targets):
        predictions = 4*tf.nn.tanh(model(graphs).globals)
        loss = loss_fn(targets, predictions)

        return predictions, loss

    train_iter = data_gen_train.generator()
    val_iter = data_gen_val.generator()
    curr_loss = 1e5
    for e in range(epochs):

        print('\n\nStarting epoch: {}'.format(e))

        training_loss = []
        val_loss = []

        # Train
        print('Training...')
        i = 1
        for graph_data_tr, targets_tr in get_batch(train_iter):
            start = time.time()
            #if i==1:
            losses_tr = train_step(graph_data_tr, targets_tr)
            end = time.time()

            training_loss.append(losses_tr.numpy())

            print('Iter: {:03d}, Tr_loss_curr: {:.4f}, Tr_loss_mean: {:.4f}'. \
                  format(i, training_loss[-1], np.mean(training_loss)), end='\t')
            print('Took {:.3f} secs'.format(end-start))
            
            i += 1 
    
        train_iter = data_gen_train.generator()
        training_loss_epoch.append(training_loss)

        # validate
        print('\nValidation...')
        i = 1
        all_targets = []
        all_outputs = []
        for graph_data_val, targets_val in get_batch(val_iter):
            start = time.time()
            outputs_val, losses_val = val_step(graph_data_val, targets_val)
            end = time.time()

            val_loss.append(losses_val.numpy())
            all_targets = np.append(all_targets, targets_val)
            all_outputs = np.append(all_outputs, outputs_val.numpy())

            print('Iter: {:03d}, Val_loss_curr: {:.4f}, Val_loss_mean: {:.4f}'. \
                  format(i, val_loss[-1], np.mean(val_loss)), end='\t')
            print('Took {:.3f} secs'.format(end-start))
            
            i += 1 
        
        val_iter = data_gen_val.generator()
        val_loss_epoch.append(val_loss)
    
        np.savez(save_dir+'/losses', training=training_loss_epoch, validation=val_loss_epoch)
    
        if np.mean(val_loss)<curr_loss:
            print('\nLoss decreased from {:.4f} to {:.4f}'.format(curr_loss, np.mean(val_loss)))
            print('Checkpointing and saving predictions to:\n{}'.format(save_dir))
            curr_loss = np.mean(val_loss)
            np.savez(save_dir+'/predictions', 
                    targets=10**all_targets, 
                    outputs=10**all_outputs)
            checkpoint.save(checkpoint_prefix)
        else:
            print('\nLoss {:.4f} greater than best loss {:.4f}'.format(np.mean(val_loss), curr_loss))
