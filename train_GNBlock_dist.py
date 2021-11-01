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

    epochs = 20
    batch_size = 1024

    data_gen_train = MPGraphDataGenerator(file_list=pion_files[10:20],
                                        cellGeo_file=data_dir+'graph_examples/cell_geo.root',
                                        batch_size=batch_size,
                                        preprocess=False,
                                        output_dir='preprocessed_data/'+particle_type+'/train/',
                                        shuffle=False)

    data_gen_val = MPGraphDataGenerator(file_list=pion_files[20:22],
                                      cellGeo_file=data_dir+'graph_examples/cell_geo.root',
                                      batch_size=batch_size,
                                      preprocess=False,
                                      output_dir='preprocessed_data/'+particle_type+'/val/',
                                      shuffle=False)

    mirrored_strategy = tf.distribute.MirroredStrategy()

    # Optimizer.
    learning_rate = 1e-3
    with mirrored_strategy.scope():
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

    repl_batches = []
    def repl_batch_fn(context):
        return repl_batches[context.replica_id_in_sync_group]

    def get_dist_batch(data_iter):
        global repl_batches

        while True:
            repl_batches = []

            for i in range(mirrored_strategy.num_replicas_in_sync):
                try:
                    graphs, targets = next(data_iter)

                    graphs = convert_to_tuple(graphs)
                    targets = tf.convert_to_tensor(targets)

                    repl_batches.append((graphs, targets))
                except Exception as e:
                    if i == 0:
                        return

                    repl_batches.append(repl_batches[-1])

            dist_batch = mirrored_strategy.experimental_distribute_values_from_function(repl_batch_fn)

            yield dist_batch
    
    loss_fn = tf.keras.losses.MeanAbsoluteError(reduction=tf.keras.losses.Reduction.NONE)

    def compute_loss(targets, predictions):
        per_example_loss = tf.reshape(tf.math.reduce_mean(loss_fn(targets, predictions)), [1])
        return tf.nn.compute_average_loss(per_example_loss)

    def train_step(dist_inputs):
        graphs, targets = dist_inputs

        with tf.GradientTape() as tape:
            predictions = model(graphs).globals
            loss = compute_loss(targets, predictions)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return loss

    @tf.function(experimental_relax_shapes=True)
    def dist_train_step(dist_inputs):
        per_replica_losses = mirrored_strategy.run(train_step, args=(dist_inputs,))
        return mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, per_replica_losses, axis=None)

    def val_step(dist_inputs):
        graphs, targets = dist_inputs

        predictions = model(graphs).globals
        loss = compute_loss(targets, predictions)

        return predictions, targets, loss

    @tf.function(experimental_relax_shapes=True)
    def dist_val_step(dist_inputs):
        repl_pred, repl_targ, repl_loss = mirrored_strategy.run(val_step, args=(dist_inputs,))
        predictions = mirrored_strategy.gather(repl_pred, axis=0)
        targets = mirrored_strategy.gather(repl_targ, axis=0)
        loss = mirrored_strategy.reduce(tf.distribute.ReduceOp.SUM, repl_loss, axis=None)

        return predictions, targets, loss

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
        for dist_input in get_dist_batch(train_iter):
            start = time.time()
            #if i==1:
            losses_tr = dist_train_step(dist_input)
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
        for dist_input in get_dist_batch(val_iter):
            start = time.time()
            outputs_val, targets_val, losses_val = dist_val_step(dist_input)
            end = time.time()

            val_loss.append(losses_val.numpy())
            all_targets = np.append(all_targets, targets_val.numpy())
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

