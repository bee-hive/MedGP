import os


def get_sample_num(data_dir, cohort, feature_list, pid):
	datapt_num = 0
	for fidx in feature_list:
		data_file = os.path.join(data_dir, cohort, '{}'.format(pid), 'feature{}.txt'.format(fidx))
		f = open(data_file, 'r')
		feature_datapt_num = int(float(f.readlines()[0]))
		f.close()
		datapt_num += feature_datapt_num
	return datapt_num