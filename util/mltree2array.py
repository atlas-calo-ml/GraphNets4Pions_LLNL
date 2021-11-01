#! /usr/bin/env python
import numpy as np

def load_tree(files, tree, branches, nmax = -1, selection=''):
  """ Load specified branches from the input TTrees into memory"""

  import ROOT
  ROOT.PyConfig.IgnoreCommandLineOptions = True
  ROOT.gROOT.SetBatch(True)
  chain = ROOT.TChain(tree)
  for f in files: chain.Add(f)

  from root_numpy import tree2array
  return tree2array(chain, branches = branches, selection = selection, start = 0, stop = nmax)

def preprocess(clusters, branches, flatten = False, label = 0):
  """ Pre-processing of the CaloML image dataset """

  ncl = len(clusters)
  nbr = len(branches)

  # one image for each layer of the calorimeter
  data = {
     # EM barrel
    'EMB1': np.zeros((ncl, 128, 4)),
    'EMB2': np.zeros((ncl, 16, 16)),
    'EMB3': np.zeros((ncl, 8, 16)),
    # TileCal barrel
    'TileBar0': np.zeros((ncl, 4, 4)),
    'TileBar1': np.zeros((ncl, 4, 4)),
    'TileBar2': np.zeros((ncl, 2, 4))
  }

  # supplemental info about clusters and cells (clusE, clusPt, nCells,...)
  for br in branches[6:]:
    data[br] = np.zeros(ncl)

  # fill the image arrays and the supplemental info
  for i in xrange(ncl):
    for j in xrange(nbr):
      data[branches[j]][i] = clusters[i][j]
      if flatten: data[branches[j]][i] = data[branches[j]][i].flatten()

  # add a vector of labels
  data['label'] = np.full((ncl, 1), label)

  return data

def export(data, output, compress):
  """ Export data to file """

  if compress:
    np.savez_compressed(output, **data)
  else:
    np.save(output, **data)

if __name__ == "__main__":

  default_branches = ['EMB1', 'EMB2', 'EMB3', 'TileBar0', 'TileBar1', 'TileBar2', 'clusterE', 'clusterPt', 'clusterEta', 'clusterPhi', 'cluster_nCells', 'cluster_sumCellE', 'cluster_emProb']
#  default_branches = ['EMB1', 'EMB2', 'EMB3', 'TileBar0', 'TileBar1', 'TileBar2', 'clusterE', 'clusterEta', 'clusterPhi', 'cluster_nCells', 'cluster_sumCellE', 'cluster_emProb']
 # default_branches = ['EMB1', 'EMB2', 'EMB3', 'TileBar0', 'TileBar1', 'TileBar2', 'clusterE', 'clusterEta', 'clusterPhi', 'cluster_nCells', 'cluster_emProb']

  import argparse
  parser = argparse.ArgumentParser(add_help=True, description='Convert root image arrays from the MLTree package to numpy arrays.')
  parser.add_argument('files', type=str, nargs='+', metavar='<file.root>', help='ROOT files containing the outputs from the MLTree package.')
  parser.add_argument('--label', '-l', required=True, type=int, help='Label for images in input array')
  parser.add_argument('--output', '-o', required=False, type=str, help='Output file to store the images', default='images')
  parser.add_argument('--nclusters', '-n', required=False, type=int, help='Number of clusters to process', default=-1)
  parser.add_argument('--tree', required=False, type=str, help='Name of input TTree.', default='ClusterTree')
  parser.add_argument('--branches', required=False, type=str, nargs='+', help='ROOT files containing the outputs from the MLTree package.', default = default_branches)
  parser.add_argument('--compress', '-c', required=False, action='store_true', help='Compress output arrays.', default=False)
  parser.add_argument('--flatten', required=False, action='store_true', help='Flatten output arrays', default=False)
  args = parser.parse_args()

  print("loading data from tree...")
  clusters = load_tree(args.files, args.tree, args.branches, args.nclusters, 'clusterE > 100')
  print("pre-processing data...")
  data = preprocess(clusters, args.branches, args.flatten, args.label)
  print("saving data...")
  export(data, args.output, args.compress)
  print("\nall done!")
