import os
import numpy as np
from array import array


def write_double_to_bin(filename, d_array):
	output_file = open(filename, 'wb')
	float_array = array('d', d_array)
	float_array.tofile(output_file)
	output_file.close()


def read_double_from_bin(filename):
	input_file = open(filename, 'rb')
	float_array = array('d')
	float_array.fromstring(input_file.read())
	return np.asarray(float_array)


def read_train_kernel(pan_array, kernel_dir):
    valid_pan = []
    valid_hyp = []
    for pan in pan_array:
        try:
            flag = np.loadtxt(os.path.join(kernel_dir, 'train_flag_{}.txt'.format(pan)), dtype=int)
            flag = np.atleast_1d(flag)[0]
            if(flag):
                hyp = read_double_from_bin(os.path.join(kernel_dir, 'train_hyp_{}.bin'.format(pan)))
                valid_pan.append(pan)
                valid_hyp.append(hyp)
        except:
            continue
    valid_pan = np.asarray(valid_pan)
    valid_hyp = np.asarray(valid_hyp)
    return valid_pan, valid_hyp


def load_ts_data(filename):
    array = np.loadtxt(filename, dtype=float)
    array = array[1:]
    t = array[0::2]
    v = array[1::2]
    return t, v

def read_one_test_data(test_dir, test_mode, pan):
    test_prefix = 'test_{}'.format(test_mode)
    pan_id = '{}'.format(pan)
    flag_file = os.path.join(test_dir, '{}_flag_{}.txt'.format(test_prefix, pan_id))
    flag_test = np.loadtxt(flag_file, dtype=int)
    flag_test = np.atleast_1d(flag_test)[0]

    if(flag_test):
        feature_file = os.path.join(test_dir, '{}_feature_{}.txt'.format(test_prefix, pan_id))
        c_feature = np.loadtxt(feature_file, dtype=int)

        pred_file = os.path.join(test_dir, '{}_pred_{}.bin'.format(test_prefix, pan_id))
        c_norm_pred = read_double_from_bin(pred_file)

        ci_file = os.path.join(test_dir, '{}_ci_{}.txt'.format(test_prefix, pan_id))
        c_ci_pred = np.loadtxt(ci_file, dtype=int)
        return flag_test, (c_feature, c_norm_pred, c_ci_pred)
    else:
        return flag_test, None
