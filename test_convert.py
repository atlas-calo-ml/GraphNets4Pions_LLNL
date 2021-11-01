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
