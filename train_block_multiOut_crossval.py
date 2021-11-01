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

from modules.mpdatagen import MPGraphDataGenerator
import modules.multiOutBlock as models
sns.set_context('poster')

if __name__=="__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/default.yaml')
    args = parser.parse_args()

    config = yaml.load(open(args.config))

    data_config = config['data']
    model_config = config['model']
    train_config = config['training']

    data_dir = data_config['data_dir']
    num_train_files = data_config['num_train_files']
    batch_size = data_config['batch_size']
    shuffle = data_config['shuffle']
    num_procs = data_config['num_procs']
    preprocess = data_config['preprocess']
    output_dir = data_config['output_dir']
    already_preprocessed = data_config['already_preprocessed']
    num_folds = data_config['num_folds']
    fold_ind = data_config['fold_ind']

    concat_input = model_config['concat_input']

    epochs = train_config['epochs']
    learning_rate = train_config['learning_rate']
    alpha = train_config['alpha']
    os.environ['CUDA_VISIBLE_DEVICES'] = str(train_config['gpu'])
    log_freq = train_config['log_freq']
    save_dir = train_config['save_dir'] + '/Block_' + args.config.replace('.yaml','').split('/')[-1]
    os.makedirs(save_dir, exist_ok=True)
    yaml.dump(config, open(save_dir + '/config.yaml', 'w'))

    logging.basicConfig(level=logging.INFO, 
                        format='%(message)s', 
                        filename=save_dir + '/output.log')
    # out_file = open(save_dir+'/output.log', 'a+')
    logging.info('Using config file {}'.format(args.config))
    # logging.info('Running training for {} with concant_input: {}\n'.format(particle_type, concat_input))

    pi0_files = np.sort(glob.glob(data_dir+'*graphs.v01*/*pi0*/*root'))
    pion_files = np.sort(glob.glob(data_dir+'*graphs.v01*/*pion*/*root'))
    train_start = 10
    train_end = train_start + num_train_files
    pi0_train_files = pi0_files[train_start:train_end]
    pion_train_files = pion_files[train_start:train_end]

    train_output_dir = None
    val_output_dir = None
            
    # Get Data
    if preprocess:
        train_output_dir = output_dir + '/train/'
        val_output_dir = output_dir + '/val/'

        if already_preprocessed:
            train_files = np.sort(glob.glob(train_output_dir+'*.p'))[:num_train_files].tolist()
            val_files = []

            start_ind = fold_ind * len(train_files) // num_folds
            end_ind = min(len(train_files), (fold_ind + 1) * len(train_files) // num_folds)

            val_files = train_files[start_ind:end_ind]
            del train_files[start_ind:end_ind]

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

    model = models.MultiOutBlockModel(global_output_size=1, num_outputs=2, model_config=model_config)

    training_loss_epoch = []
    training_loss_regress_epoch = []
    training_loss_class_epoch = []
    val_loss_epoch = []
    val_loss_regress_epoch = []
    val_loss_class_epoch = []
    
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
        regress_loss = mae_loss(targets[:,:1], predictions[0])
        class_loss = bce_loss(targets[:,1:], predictions[1])
        combined_loss = alpha*regress_loss + (1 - alpha)*class_loss 
        return regress_loss, class_loss, combined_loss

    @tf.function(input_signature=[graph_spec, tf.TensorSpec(shape=[None,2], dtype=tf.float32)])
    def train_step(graphs, targets):
        with tf.GradientTape() as tape:
            outputs = model(graphs)
            predictions = [outputs[0].globals, outputs[1].globals]
            regress_loss, class_loss, loss = loss_fn(targets, predictions)
        
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return regress_loss, class_loss, loss

    @tf.function(input_signature=[graph_spec, tf.TensorSpec(shape=[None,2], dtype=tf.float32)])
    def val_step(graphs, targets):
        outputs = model(graphs)
        predictions = [outputs[0].globals, outputs[1].globals]
        regress_loss, class_loss, loss = loss_fn(targets, predictions)

        return regress_loss, class_loss, loss, predictions

    curr_loss = 1e5
    for e in range(epochs):

        logging.info('\nStarting epoch: {}'.format(e))

        training_loss = []
        training_loss_regress = []
        training_loss_class = []
        val_loss = []
        val_loss_regress = []
        val_loss_class = []

        # Train
        logging.info('Training...')
        i = 1
        for graph_data_tr, targets_tr in get_batch(data_gen_train.generator()):#train_iter):
            start = time.time()
            #if i==1:
            losses_tr_rg, losses_tr_cl, losses_tr = train_step(graph_data_tr, targets_tr)
            end = time.time()

            training_loss.append(losses_tr.numpy())
            training_loss_regress.append(losses_tr_rg.numpy())
            training_loss_class.append(losses_tr_cl.numpy())

            if not (i-1)%log_freq:
                logging.info('Iter: {:04d}, Tr_loss_curr: {:.4f}, Tr_loss_mean: {:.4f}, Tr_loss_rg_curr: {:.4f}, Tr_loss_rg_mean: {:.4f}, Tr_loss_cl_curr: {:.4f}, Tr_loss_cl_mean: {:.4f}, Took {:.3f}secs'. \
                      format(i, 
                             training_loss[-1], np.mean(training_loss), 
                             training_loss_regress[-1], np.mean(training_loss_regress), 
                             training_loss_class[-1], np.mean(training_loss_class), 
                             end-start))
                # logging.info('Took {:.3f} secs'.format(end-start))
            
            i += 1 

        training_loss_epoch.append(training_loss)
        training_loss_regress_epoch.append(training_loss_regress)
        training_loss_class_epoch.append(training_loss_class)

        # validate
        logging.info('\nValidation...')
        i = 1
        all_targets = []
        all_outputs = []
        for graph_data_val, targets_val in get_batch(data_gen_val.generator()):#val_iter):
            start = time.time()
            losses_val_rg, losses_val_cl, losses_val, output_vals = val_step(graph_data_val, targets_val)
            end = time.time()

            targets_val = targets_val.numpy()
            output_vals_0 = output_vals[0].numpy()
            output_vals_1 = output_vals[1].numpy()

            targets_val[:,0] = 10**targets_val[:,0]
            output_vals_0 = 10**output_vals_0
            # targets_val[:,1] = 1 / (1 + np.exp(targets_val[:,1]))
            output_vals_1 =  tf.math.sigmoid(output_vals_1)   # 1 / (1 + np.exp(output_vals_1))

            output_vals = np.hstack([output_vals_0, output_vals_1])

            val_loss.append(losses_val.numpy())
            val_loss_regress.append(losses_val_rg.numpy())
            val_loss_class.append(losses_val_cl.numpy())

            all_targets.append(targets_val)
            all_outputs.append(output_vals)

            if not (i-1)%log_freq:
                logging.info('Iter: {:04d}, Val_loss_curr: {:.4f}, Val_loss_mean: {:.4f}, Val_loss_rg_curr: {:.4f}, Val_loss_rg_mean: {:.4f}, Val_loss_cl_curr: {:.4f}, Val_loss_cl_mean: {:.4f}, Took {:.3f}secs'. \
                      format(i, 
                             val_loss[-1], np.mean(val_loss), 
                             val_loss_regress[-1], np.mean(val_loss_regress), 
                             val_loss_class[-1], np.mean(val_loss_class), 
                             end-start))
                # logging.info('Took {:.3f} secs'.format(end-start))
            
            i += 1 

        all_targets = np.concatenate(all_targets)
        all_outputs = np.concatenate(all_outputs)
        
        val_loss_epoch.append(val_loss)
        val_loss_regress_epoch.append(val_loss_regress)
        val_loss_class_epoch.append(val_loss_class)
    
        np.savez(save_dir+'/losses', 
                training=training_loss_epoch, validation=val_loss_epoch,
                training_regress=training_loss_regress_epoch, validation_regress=val_loss_regress_epoch,
                training_class=training_loss_class_epoch, validation_class=val_loss_class_epoch,
                )
        # checkpoint.save(checkpoint_prefix)
    
        if np.mean(val_loss)<curr_loss:
            logging.info('\nLoss decreased from {:.4f} to {:.4f}'.format(curr_loss, np.mean(val_loss)))
            logging.info('Checkpointing and saving predictions to:\n{}'.format(save_dir))
            curr_loss = np.mean(val_loss)
            np.savez(save_dir+'/predictions', 
                    targets=all_targets, 
                    outputs=all_outputs)
            checkpoint.save(checkpoint_prefix)

    out_file.close()
