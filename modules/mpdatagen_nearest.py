import numpy as np
import glob
import os
import uproot as ur
import time
from multiprocessing import Process, Queue, set_start_method
import compress_pickle as pickle
from scipy.stats import circmean
from sklearn.neighbors import NearestNeighbors
import random

def geo_coords_to_xyz(geo_data):
    geo_xyz = np.zeros((geo_data['cell_geo_eta'][0].shape[0], 3))
    geo_xyz[:, 0] = geo_data["cell_geo_rPerp"][0] * np.cos(geo_data["cell_geo_phi"][0])
    geo_xyz[:, 1] = geo_data["cell_geo_rPerp"][0] * np.sin(geo_data["cell_geo_phi"][0])
    cell_geo_theta = 2 * np.arctan(np.exp(-geo_data["cell_geo_eta"][0]))
    geo_xyz[:, 2] = geo_data["cell_geo_rPerp"][0]/np.tan(cell_geo_theta)
    return geo_xyz


class MPGraphDataGeneratorMultiOut:
    """DataGenerator class for extracting and formating data from list of root files"""
    def __init__(self,
                 pi0_file_list: list,
                 pion_file_list: list,
                 cellGeo_file: str,
                 batch_size: int,
                 k: int,
                 use_xyz: bool = True,
                 shuffle: bool = True,
                 num_procs: int = 32,
                 preprocess: bool = False,
                 output_dir: str = None):
        """Initialization"""

        self.preprocess = preprocess
        self.output_dir = output_dir

        if self.preprocess and self.output_dir is not None:
            self.pi0_file_list = pi0_file_list
            self.pion_file_list = pion_file_list
            assert len(pi0_file_list) == len(pion_file_list)
            self.num_files = len(self.pi0_file_list)
        else:
            self.file_list = pi0_file_list
            self.num_files = len(self.file_list)
        
        self.cellGeo_file = cellGeo_file
        
        self.cellGeo_data = ur.open(self.cellGeo_file)['CellGeo']
        self.geoFeatureNames = self.cellGeo_data.keys()[1:9]
        self.nodeFeatureNames = ['cluster_cell_E', *self.geoFeatureNames[:-2]]
        self.edgeFeatureNames = self.cellGeo_data.keys()[9:]
        self.num_nodeFeatures = len(self.nodeFeatureNames)
        self.num_edgeFeatures = len(self.edgeFeatureNames)

        self.cellGeo_data = self.cellGeo_data.arrays(library='np')
        self.geo_xyz = geo_coords_to_xyz(self.cellGeo_data)
        self.geo_xyz /= np.max(self.geo_xyz)
        self.use_xyz = use_xyz
        self.cellGeo_ID = self.cellGeo_data['cell_geo_ID'][0]
        self.sorter = np.argsort(self.cellGeo_ID)
        
        self.batch_size = batch_size
        self.k = k
        self.shuffle = shuffle
        
        if self.shuffle: np.random.shuffle(self.file_list)
        
        self.num_procs = num_procs
        self.procs = []

        if self.preprocess and self.output_dir is not None:
            os.makedirs(self.output_dir, exist_ok=True)
            self.preprocess_data()

    def get_cluster_calib(self, event_data, event_ind, cluster_ind):
        """ Reading cluster calibration energy """ 
            
        cluster_calib_E = event_data['cluster_ENG_CALIB_TOT'][event_ind][cluster_ind]

        if cluster_calib_E <= 0:
            return None

        return np.log10(cluster_calib_E)
            
    def get_data(self, event_data, event_ind, cluster_ind):
        """ Reading Node features """ 

        cell_IDs = event_data['cluster_cell_ID'][event_ind][cluster_ind]
        cell_IDmap = self.sorter[np.searchsorted(self.cellGeo_ID, cell_IDs, sorter=self.sorter)]
        
        nodes = np.log10(event_data['cluster_cell_E'][event_ind][cluster_ind])
        global_node = np.log10(event_data['cluster_E'][event_ind][cluster_ind])
        
        # Scaling the cell_geo_sampling by 28
        nodes = np.append(nodes, self.cellGeo_data['cell_geo_sampling'][0][cell_IDmap]/28.)
        for f in self.nodeFeatureNames[2:4]:
            nodes = np.append(nodes, self.cellGeo_data[f][0][cell_IDmap])
        # Scaling the cell_geo_rPerp by 3000
        nodes = np.append(nodes, self.cellGeo_data['cell_geo_rPerp'][0][cell_IDmap]/3000.)
        for f in self.nodeFeatureNames[5:]:
            nodes = np.append(nodes, self.cellGeo_data[f][0][cell_IDmap])

        nodes = np.reshape(nodes, (len(self.nodeFeatureNames), -1)).T
        cluster_num_nodes = len(nodes)
       
        # Using kNN on eta, phi, rPerp for creating graph
        curr_k = np.min([self.k, nodes.shape[0]])
        if self.use_xyz:
            nodes_NN_feats = self.geo_xyz[cell_IDmap, :]
        else:
            nodes_NN_feats = nodes[:, 2:5]

        # nbrs = NearestNeighbors(n_neighbors=curr_k, algorithm='ball_tree').fit(nodes[:, 2:5])
        nbrs = NearestNeighbors(n_neighbors=curr_k, algorithm='ball_tree').fit(nodes_NN_feats)
        distances, indices = nbrs.kneighbors(nodes_NN_feats)
        
        senders = indices[:, 1:].flatten()
        receivers = np.repeat(indices[:, 0], curr_k-1)
        edges = distances[:, 1:].reshape(-1, 1)

        return nodes, np.array([global_node]), senders, receivers, edges
    
    def preprocessor(self, worker_id):
        file_num = worker_id
        while file_num < self.num_files:
            file = self.pion_file_list[file_num]
            event_tree = ur.open(file)['EventTree']
            num_events = event_tree.num_entries

            event_data = event_tree.arrays(library='np')

            preprocessed_data = []

            for event_ind in range(num_events):
                num_clusters = event_data['nCluster'][event_ind]
                
                for i in range(num_clusters):
                    cluster_calib_E = self.get_cluster_calib(event_data, event_ind, i)
                    
                    if cluster_calib_E is None:
                        continue
                        
                    nodes, global_node, senders, receivers, edges = self.get_data(event_data, event_ind, i)

                    graph = {'nodes': nodes.astype(np.float32), 'globals': global_node.astype(np.float32),
                        'senders': senders.astype(np.int32), 'receivers': receivers.astype(np.int32),
                        'edges': edges.astype(np.float32)}
                    target = np.reshape([cluster_calib_E.astype(np.float32), 1], [1,2])

                    preprocessed_data.append((graph, target))

            file = self.pi0_file_list[file_num]
            event_tree = ur.open(file)['EventTree']
            num_events = event_tree.num_entries

            event_data = event_tree.arrays(library='np')

            for event_ind in range(num_events):
                num_clusters = event_data['nCluster'][event_ind]
                
                for i in range(num_clusters):
                    cluster_calib_E = self.get_cluster_calib(event_data, event_ind, i)
                    
                    if cluster_calib_E is None:
                        continue
                        
                    nodes, global_node, senders, receivers, edges = self.get_data(event_data, event_ind, i)

                    graph = {'nodes': nodes.astype(np.float32), 'globals': global_node.astype(np.float32),
                        'senders': senders.astype(np.int32), 'receivers': receivers.astype(np.int32),
                        'edges': edges.astype(np.float32)}
                    target = np.reshape([cluster_calib_E.astype(np.float32), 0], [1,2])

                    preprocessed_data.append((graph, target))

            random.shuffle(preprocessed_data)

            pickle.dump(preprocessed_data, open(self.output_dir + 'data_{}.p'.format(file_num), 'wb'), compression='gzip')

            file_num += self.num_procs

    def preprocess_data(self):
        print('\nPreprocessing and saving data to {}'.format(self.output_dir))
        for i in range(self.num_procs):
            p = Process(target=self.preprocessor, args=(i,), daemon=True)
            p.start()
            self.procs.append(p)
        
        for p in self.procs:
            p.join()

        self.file_list = [self.output_dir + 'data_{}.p'.format(i) for i in range(self.num_files)]

    def preprocessed_worker(self, worker_id, batch_queue):
        batch_graphs = []
        batch_targets = []

        file_num = worker_id
        while file_num < self.num_files:
            file_data = pickle.load(open(self.file_list[file_num], 'rb'), compression='gzip')

            for i in range(len(file_data)):
                batch_graphs.append(file_data[i][0])
                batch_targets.append(file_data[i][1])
                    
                if len(batch_graphs) == self.batch_size:
                    batch_targets = np.reshape(np.array(batch_targets), [-1,2]).astype(np.float32)
                    
                    batch_queue.put((batch_graphs, batch_targets))
                    
                    batch_graphs = []
                    batch_targets = []

            file_num += self.num_procs
                    
        if len(batch_graphs) > 0:
            batch_targets = np.reshape(np.array(batch_targets), [-1,2]).astype(np.float32)
            
            batch_queue.put((batch_graphs, batch_targets))

    def worker(self, worker_id, batch_queue):
        if self.preprocess:
            self.preprocessed_worker(worker_id, batch_queue)
        else:
            raise Exception('Preprocessing is required for combined classification/regression models.')
        
    def check_procs(self):
        for p in self.procs:
            if p.is_alive(): return True
        
        return False

    def kill_procs(self):
        for p in self.procs:
            p.kill()

        self.procs = []
    
    def generator(self):
        # for file in self.file_list:
        batch_queue = Queue(2 * self.num_procs)
            
        for i in range(self.num_procs):
            p = Process(target=self.worker, args=(i, batch_queue), daemon=True)
            p.start()
            self.procs.append(p)
        
        while self.check_procs() or not batch_queue.empty():
            try:
                batch = batch_queue.get(True, 0.0001)
            except:
                continue
            
            yield batch
        
        for p in self.procs:
            p.join()
            
        
if __name__ == '__main__':
    data_dir = '/usr/workspace/pierfied/preprocessed/data/'
    out_dir = '/usr/workspace/pierfied/preprocessed/preprocessed_data/'
    pion_files = np.sort(glob.glob(data_dir+'user*.root'))
    
    data_gen = MPGraphDataGenerator(file_list=pion_files, 
                                  cellGeo_file=data_dir+'cell_geo.root',
                                  batch_size=32,
                                  shuffle=False,
                                  num_procs=32,
                                  preprocess=True,
                                  output_dir=out_dir)

    gen = data_gen.generator()
    
    from tqdm.auto import tqdm
     
    for batch in tqdm(gen):
        pass
        
    exit()
