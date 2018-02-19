import os
import json
import numpy as np
from ..util.binaryIO import read_train_kernel
from ..visualization.printkernel import print_kernel_info
from .cluster import run_clustering_top
from .feature_extraction import extract_kernel_feature
from .mode_estimate import output_mode_kernel


def kernel_clustering_top(exp_config, fold=-1, algorithm=None, figure_plot_mode=1, figure_config=None):
    # parse json file and get parameters for the experiments
    exp_param = json.load(open(exp_config, 'r'))
    if((figure_config is not None) and (figure_config != 'None')):
        plot_param = json.load(open(figure_config, 'r'))
    else:
        plot_param = None

    # decide patients to include in this fold (-1 for all)
    cv_assign = np.loadtxt(os.path.join(exp_param["cv_assign_file"]), dtype=int)
    valid_pan = np.genfromtxt(os.path.join(exp_param["data_dir"], exp_param["cohort_id_list"]), dtype=str)
    if(fold != -1):
        valid_pan = valid_pan[np.where(cv_assign != fold)]
    print('Info: # of ids available for this fold ({}): {}'.format(fold, len(valid_pan)))

    # read in successfully trained kernels
    kernel_pan, kernel_hyp = read_train_kernel(pan_array=valid_pan, kernel_dir=exp_param["exp_train_dir"])
    if(len(kernel_pan) != len(valid_pan)):
        print('Warning: # of valid trained subjects ({}) less than expected ({})'.format(
            len(kernel_pan), len(valid_pan))
        )
    else:
        print('Info: successfully load all ids ({})'.format(len(kernel_pan)))

    # extract features of each kernel components
    # if the kernel is univariate, each component is same as the kernel per subject
    comp_pan, comp_qidx, comp_feature = extract_kernel_feature(
                                            kernel_type=exp_param["kernel"],
                                            Q=exp_param["Q"], D=exp_param["D"], R=exp_param["R"],
                                            pan_array=kernel_pan, hyp_array=kernel_hyp)

    # do kernel clustering
    comp_cluster_num, comp_cluster_assign = run_clustering_top(algorithm=algorithm,
                                                                feature=comp_feature,
                                                                max_cluster_num=exp_param["Q"])

    # estimate mode kernel using KDE and output parameters for medgpc testing
    mode_hyp = output_mode_kernel(fold=fold, exp_param=exp_param,
                                  pan_array=kernel_pan, hyp_array=kernel_hyp,
                                  mixture_pan=comp_pan, mixture_index=comp_qidx,
                                  mixture_cluster_num=comp_cluster_num, 
                                  mixture_cluster_assign=comp_cluster_assign,
                                  kernclust_alg=algorithm,
                                  plotting_mode=figure_plot_mode,
                                  plotting_param=plot_param)
    print('mode kernel info:')
    print_kernel_info(kernel_type=exp_param["kernel"], hyp=mode_hyp,
                      Q=comp_cluster_num, D=exp_param["D"], R=exp_param["R"])
    
