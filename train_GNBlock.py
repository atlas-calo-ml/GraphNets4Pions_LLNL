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
import sonnet as snt

from modules.graph_data import GraphDataGenerator
import modules.gnBlock as models
sns.set_context('poster')



if __name__=="__main__":
    
    concat_input = sys.argv[1]
    print('Running training concant_input: {}\n'.format(concat_input))
            
    # Get Data
    data_dir = '/usr/workspace/hip/ML4Jets/regression_images/'
    pion_files = np.sort(glob.glob(data_dir+'*graphs.v01*/*singlepion*/*root'))

    data_gen_train = GraphDataGenerator(file_list=pion_files[10:20],
                                        cellGeo_file=data_dir+'graph_examples/cell_geo.root',
                                        batch_size=32,
                                        shuffle=False)

    data_gen_val = GraphDataGenerator(file_list=pion_files[20:22],
                                      cellGeo_file=data_dir+'graph_examples/cell_geo.root',
                                      batch_size=32,
                                      shuffle=False)

    # Optimizer.
    learning_rate = 1e-2
    optimizer = snt.optimizers.Adam(learning_rate)

    model = models.GNBlockModel(global_output_size=1, concat_input=concat_input)
    batch_size = 1024
    epochs = 200

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

    def get_batch(data_gen, batch_size=32):
        graph_data = []
        targets = []
        end_batch = False
        for i in range(batch_size):
            g, t = next(data_gen.gen)
            graph_data.append(g)
            targets.append(t)
            if data_gen.end:
                end_batch = True
                break

        graph_data = utils_tf.data_dicts_to_graphs_tuple(graph_data)
        return graph_data, targets

    def create_loss(targets, output):
        output_globals = 4*tf.nn.tanh(output)
        loss = tf.keras.losses.mae(targets, output_globals)
        return loss


    # Training.
    def update_step(inputs_tr, targets_tr):
        with tf.GradientTape() as tape:
            outputs_tr = model(inputs_tr)
            # Loss.
            loss_tr = create_loss(targets_tr, outputs_tr.globals)
            loss_tr = tf.math.reduce_mean(loss_tr)

        gradients = tape.gradient(loss_tr, model.trainable_variables)
        optimizer.apply(gradients, model.trainable_variables)
        return outputs_tr, loss_tr, gradients

    def val_step(inputs, targets):
        outputs = model(inputs)
        # Loss.
        loss = create_loss(targets, outputs.globals)
        loss = tf.math.reduce_mean(loss)

        return outputs, loss

    for e in range(epochs):

        print('\n\nStarting epoch: {}'.format(e))

        training_loss = []
        val_loss = []

        # Train
        print('Training...')
        i = 1
        while not data_gen_train.end:
            start = time.time()
            if i==1:
                graph_data_tr, targets_tr = get_batch(data_gen_train, batch_size)
            outputs_tr, losses_tr, _ = update_step(graph_data_tr, targets_tr)
            end = time.time()

            training_loss.append(losses_tr.numpy())

            print('Iter: {:03d}, Tr_loss_curr: {:.4f}, Tr_loss_mean: {:.4f}'. \
                  format(i, training_loss[-1], np.mean(training_loss)), end='\t')
            print('Took {:.3f} secs'.format(end-start))
            
            i += 1 
    
        training_loss_epoch.append(training_loss)
        data_gen_train.restart()

        # validate
        print('\nValidation...')
        i = 1
        while not data_gen_val.end:
            graph_data_val, targets_val = get_batch(data_gen_val, batch_size)
            
            start = time.time()
            outputs_val, losses_val = val_step(graph_data_val, targets_val)
            end = time.time()

            val_loss.append(losses_val.numpy())

            print('Iter: {:03d}, Val_loss_curr: {:.4f}, Val_loss_mean: {:.4f}'. \
                  format(i, val_loss[-1], np.mean(val_loss)), end='\t')
            print('Took {:.3f} secs'.format(end-start))
            
            i += 1 
        
        val_loss_epoch.append(val_loss)
        data_gen_val.restart()
    
        np.savez(save_dir+'/losses', training=training_loss_epoch, validation=val_loss_epoch)
        checkpoint.save(checkpoint_prefix)
    

