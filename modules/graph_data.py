import numpy as np
import glob
import os
import uproot as ur
import time

class GraphDataGenerator():
    """DataGenerator class for extracting and formating data from list of root files"""
    def __init__(self,
                 file_list: list,
                 cellGeo_file: str,
                 batch_size: int,
                 shuffle: bool = True):
        """Initialization"""
        self.file_list = file_list
        self.num_files = len(self.file_list)
        
        self.cellGeo_file = cellGeo_file
        
        self.cellGeo_data = ur.open(self.cellGeo_file)['CellGeo']
        self.geoFeatureNames = self.cellGeo_data.keys()[1:9]
        self.nodeFeatureNames = ['cluster_cell_E', *self.geoFeatureNames[:-2]]
        self.edgeFeatureNames = self.cellGeo_data.keys()[9:]
        self.num_nodeFeatures = len(self.nodeFeatureNames)
        self.num_edgeFeatures = len(self.edgeFeatureNames)
        
        self.cellGeo_data = self.cellGeo_data.arrays(library='np')
        self.cellGeo_ID = self.cellGeo_data['cell_geo_ID'][0]
        self.sorter = np.argsort(self.cellGeo_ID)
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        
        if self.shuffle: np.random.shuffle(self.file_list)
        
        self.file_ind = 0
        self.num_events = None
        self.reset()
        
        self.start_tree = time.time()
        self.load_eventTree()
        self.update_event()
        self.gen = self.yield_cluster()
        self.end = False
        
        return 
        
    def load_eventTree(self):
        """ Loads one root files into memory """
        self.end_tree = time.time()
        mins = int((self.end_tree - self.start_tree)/60)
        secs = int((self.end_tree - self.start_tree)%60)
        # print('\n\nReading all events in tree took {:d} mins, {:d} secs'.format(mins, secs))

        # print('\n\nFile ind: {}, Loading EventTree...'.format(self.file_ind))
        start = time.time()
        self.eventTree_data = ur.open(self.file_list[self.file_ind])['EventTree'].arrays(library='np')
        end = time.time()
        mins = int((end-start)/60)
        secs = int((end-start)%60)
        # print('Took {:d} mins, {:d} secs\n\n'.format(mins, secs))
        self.num_events = self.eventTree_data['eventNumber'].shape[0]
        self.file_ind += 1
        self.start_tree = time.time()
        return
        
    def yield_cluster(self):
        """ Generates data one cluster at a time, indefinitely """
        while not self.end:
            start = time.time()
            cluster_calib_E = self.get_cluster_calib()
            nodes, global_node = self.get_nodes()
            end = time.time()
            node_time = 1000*(end - start)
            start = time.time()
            senders, receivers, edges = self.get_edges()
            end = time.time()
            edge_time = 1000*(end - start) 
            # print('\tCluster: {}, Nodes: {}, Senders: {}, Recievers: {}, Edges: {}, CalibE: {:.3f}, CLusterE: {:.3f}, Nodes took: {:.3f}ms, Edges took: {:.3f}ms'.\
            #      format(self.event_cluster_ind, nodes.shape, senders.shape, receivers.shape, edges.shape,
            #             cluster_calib_E, global_node[0], node_time, edge_time))
            self.update_all()
            yield {'nodes': nodes.astype(np.float32), 'globals': global_node.astype(np.float32), 
                   'senders': senders, 'receivers': receivers, 'edges': edges.astype(np.float32)}, cluster_calib_E
        return
   
    def get_cluster_calib(self):
        """ Reading cluster calibration energy """ 

        while not self.event_nClusters:
            self.update_event()
            
        cluster_calib_E = self.eventTree_data['cluster_ENG_CALIB_TOT'][self.event_ind][self.event_cluster_ind]

        while not cluster_calib_E>0:
            # print('\tCluster Calib is 0, skipping...')
            self.update_all()
            cluster_calib_E = self.eventTree_data['cluster_ENG_CALIB_TOT'][self.event_ind][self.event_cluster_ind]

        return np.log10(cluster_calib_E)
            
    def get_nodes(self):
        """ Reading Node features """ 

        self.cell_IDs = self.eventTree_data['cluster_cell_ID'][self.event_ind][self.event_cluster_ind]
        self.cell_IDmap = self.sorter[np.searchsorted(self.cellGeo_ID, self.cell_IDs, sorter=self.sorter)]
        
        nodes = np.log10(self.eventTree_data['cluster_cell_E'][self.event_ind][self.event_cluster_ind])
        global_node = np.log10(self.eventTree_data['cluster_E'][self.event_ind][self.event_cluster_ind])
        
        # Scaling the cell_geo_sampling by 28
        nodes = np.append(nodes, self.cellGeo_data['cell_geo_sampling'][0][self.cell_IDmap]/28.)
        for f in self.nodeFeatureNames[2:4]:
            nodes = np.append(nodes, self.cellGeo_data[f][0][self.cell_IDmap])
        # Scaling the cell_geo_rPerp by 3000
        nodes = np.append(nodes, self.cellGeo_data['cell_geo_rPerp'][0][self.cell_IDmap]/3000.)
        for f in self.nodeFeatureNames[5:]:
            nodes = np.append(nodes, self.cellGeo_data[f][0][self.cell_IDmap])

        nodes = np.reshape(nodes, (len(self.nodeFeatureNames), -1)).T
        self.cluster_num_nodes = len(nodes)
        
        return nodes, np.array([global_node])
    
    def get_edges(self):
        """ 
        Reading edge features 
        Resturns senders, receivers, and edges    
        """ 
        edge_inds = np.zeros((self.cluster_num_nodes, self.num_edgeFeatures))
        for i, f in enumerate(self.edgeFeatureNames):
            edge_inds[:, i] = self.cellGeo_data[f][0][self.cell_IDmap]
        edge_inds[np.logical_not(np.isin(edge_inds, self.cell_IDmap))] = np.nan
        
        senders, edge_on_inds = np.isin(edge_inds, self.cell_IDmap).nonzero()
        self.cluster_num_edges = len(senders)
        edges = np.zeros((self.cluster_num_edges, self.num_edgeFeatures))
        edges[np.arange(self.cluster_num_edges), edge_on_inds] = 1
        
        cell_IDmap_sorter = np.argsort(self.cell_IDmap)
        rank = np.searchsorted(self.cell_IDmap, edge_inds , sorter=cell_IDmap_sorter)
        recievers = cell_IDmap_sorter[rank[rank!=self.cluster_num_nodes]]
        
        return senders, recievers, edges
    
    def update_all(self):
        """ 
        Go to next cluster
        Perform checks if you run out of events or files 
        """
        self.event_cluster_ind += 1
        
        # if self.event_cluster_ind == self.event_nClusters: 
        #     self.update_event()
        while self.event_cluster_ind == self.event_nClusters: 
            self.update_event()
        
        # if self.file_ind == self.num_files:
            # self.file_ind = 0
            # if self.shuffle: np.random.shuffle(self.file_list)

        return

    def update_event(self):
        """
        Go to next event
        """
        self.event_ind +=1 
        self.event_cluster_ind = 0

        if self.event_ind >= self.num_events:
            self.reset()
            if self.file_ind == self.num_files:
                self.end = True
                # print('\n\n Ran out of files')
                return
            else:
                self.load_eventTree()
                self.event_ind = 0
        
        self.event_nClusters = self.eventTree_data['nCluster'][self.event_ind]
        # print('Event index: {}, Num Clusters: {}'.format(self.event_ind, self.event_nClusters))
        return
    
    def reset(self):
        """
        Reset variables after loading a new root file
        """
        self.event_ind = -1
        self.event_nClusters = None
        self.event_cluster_ind = 0
        self.cluster_num_nodes = None
        self.cluster_num_edges = None
        return

    def restart(self):
        """
        Restart data generator after iterating through all root files
        """
        # print('\n\nRestarting...')
        self.file_ind = 0
        if self.shuffle: np.random.shuffle(self.file_list)
        self.end = False
        self.reset()
        self.load_eventTree()
        self.update_event()
        self.gen = self.yield_cluster()


if __name__=='__main__':

    data_dir = '/usr/workspace/hip/ML4Jets/regression_images/'
    pion_files = np.sort(glob.glob(data_dir+'*graphs.v01*/*singlepion*/*root'))

    data_gen = GraphDataGenerator(file_list=pion_files, 
                                  cellGeo_file=data_dir+'graph_examples/cell_geo.root',
                                  batch_size=32,
                                  shuffle=False)

    for nodes, senders, recievers, edges in data_gen.gen:
        pass
