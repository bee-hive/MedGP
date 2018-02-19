import os
import json
import numpy as np
from ..util.binaryIO import read_double_from_bin, write_double_to_bin, load_ts_data, read_one_test_data


def	eval_medgpc_top(exp_config, test_mode):
	# load configuration file
	exp_param = json.load(open(exp_config, 'r'))
	
	# evaluate testing results
	raw_data_dir = exp_param["data_dir"]
	test_dir = exp_param["exp_test_dir"]
	test_feature = [int(x) for x in exp_param["feature_index"][:-1].split(" ")]

	valid_pan = np.loadtxt(os.path.join(raw_data_dir, exp_param["cohort_id_list"]), dtype=str)

	compute_one_metric = {'mae': compute_one_mae, 'ci_ratio': compute_one_coverage}
	for metric in ['mae', 'ci_ratio']:
		for fidx in test_feature:
			carray = read_double_from_bin(os.path.join(raw_data_dir, 'feature{}_stat.bin'.format(fidx)))
			pop_mean = carray[0]
			pop_std = carray[1]
			all_pwise_metric = np.zeros(len(valid_pan))
			for i, pan in enumerate(valid_pan):	
				# check testing flag
				flag_test, (c_feature, c_norm_pred, c_ci_pred) = read_one_test_data(test_dir, test_mode, pan)
				if(flag_test):
						

						valid_idx = np.where(c_feature == fidx)
						pan_pred = c_norm_pred[valid_idx]*pop_std + pop_mean
						pan_ci_flag = c_ci_pred[valid_idx]

						raw_file = os.path.join(raw_data_dir, '{}'.format(pan), 'feature{}.txt'.format(fidx))
						raw_time, raw_val = load_ts_data(raw_file)
						
						assert(len(pan_pred) == len(raw_val))
						assert(len(pan_ci_flag) == len(raw_val))

						pan_error = raw_val - pan_pred
						all_pwise_metric[i] = compute_one_metric[metric](pan_error, pan_ci_flag)

				else:
					all_pwise_metric[i] = -1.
			# write valid feature-wise patient-wise error metrics to file
			all_pwise_metric = all_pwise_metric[np.where(all_pwise_metric >= 0.)]
			print('valid testing patients for feature{}: {}/{}'.format(fidx, len(all_pwise_metric), len(valid_pan)))
			output_file = os.path.join(test_dir, 'test_{}_feature{}_{}.bin'.format(test_mode, fidx, metric))
			write_double_to_bin(output_file, all_pwise_metric)
			print('write results to file: {}'.format(output_file))


def compute_one_mae(error_arr, coverage_flag_arr):
	one_mae = np.nanmean(np.abs(error_arr))
	return one_mae


def compute_one_coverage(error_arr, coverage_flag_arr):
	one_coverage = 100.*np.nanmean(coverage_flag_arr)
	return one_coverage

