import os
import time
import argparse
from .evals import eval_medgpc_top

if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('--exp-config', type=str, required=True, help='the json configuration file for the experiment')
	parser.add_argument('--test-mode', type=str, required=True, help='the testing mode for medgpc; choose between "mean_wo_update" or "mean_w_update"')
	args = parser.parse_args()

	exp_config = args.exp_config
	test_mode = args.test_mode

	t0 = time.time()
	eval_medgpc_top(exp_config=exp_config, test_mode=test_mode)
	tdiff = time.time()-t0
	print('Finished evaluation; elapsed time: {} seconds'.format(tdiff))

