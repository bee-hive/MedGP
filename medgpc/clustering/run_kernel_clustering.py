import os
import time
import argparse
from .kernclust import kernel_clustering_top

if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('--exp-config', type=str, required=True, help='the json configuration file for the experiment')
	parser.add_argument('--fold', type=int, default=-1, help='the index of cross-validation fold to run clustering (e.g. 0~9 for 10-fold cross-validation); use -1 to run on all available patients')
	parser.add_argument('--kernel-cluster-alg', type=str, default=None, help='the algorithm to use for kernel clustering; default is none; no need to set up if running univariate GP models')
	parser.add_argument('--figure-config', type=str, default=None, help='the json configuration file for visualizing mode kernels')
	parser.add_argument('--figure-plot-mode', type=int, default=1, help='the level of visualization; 0: no visualization, 1: visualizing only mode kernels, other positive number: visualizing all')
	args = parser.parse_args()

	exp_config = args.exp_config
	kc_algorithm = args.kernel_cluster_alg
	fold = args.fold
	figure_plot_mode = args.figure_plot_mode
	figure_config = args.figure_config

	t0 = time.time()
	kernel_clustering_top(exp_config=exp_config, fold=fold, algorithm=kc_algorithm, 
	                      figure_plot_mode=figure_plot_mode, figure_config=figure_config)
	tdiff = time.time()-t0
	print('Finished one kernel clustering job; elapsed time: {} seconds'.format(tdiff))


