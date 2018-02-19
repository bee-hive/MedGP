import os
import sys
import stat
import argparse
import json
import numpy as np

from .hpc import write_scheduler_sh
from .config import write_medgpc_bound, write_medgpc_config_json
from .profile import get_sample_num

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # path/computing/data arguments
    parser.add_argument('--path-config', type=str, required=True, help='the json file that configures path of experiments')
    parser.add_argument('--hpc-config', type=str, required=True, help='the json file that configures computing cluster scripts')
    parser.add_argument('--feature-config', type=str, required=True, help='the json file that configures the features/covariates to use')
    parser.add_argument('--figure-config', type=str, default=None, help='the json file that configures figures; will use default setup if not specified')
    parser.add_argument('--cohort', type=str, required=True, help='the cohort to use')
    parser.add_argument('--exp-prefix', type=str, default='exp_0000', help='the prefix of the experiment name')
    parser.add_argument('--cv-fold-num', type=int, default=10, help='number of cross-validation folds; default is 10')
    parser.add_argument('--cv-seed', type=int, default=718, help='random seed for create cross-validation assignments; choose the same seed to keep same assignment across experiments on the same cohort')

    # model arguments
    parser.add_argument('--kernel', type=str, default='SE', help='the kernel to use')
    parser.add_argument('--prior', type=str, default=None, help='the prior to use for the kernel ("hier-gamma" or "None")')
    parser.add_argument('--Q', type=int, default=1, help='the number of mixture kernel component for miltivariate GP models')
    parser.add_argument('--R', type=int, default=1, help='the number of rank for each mixture kernel component')
    parser.add_argument('--eta', type=float, default=0.1, help='the eta value in the hierachical gamma prior')
    parser.add_argument('--beta-lam', type=float, default=0.01, help='the beta value for the laplace prior on lambda')
    parser.add_argument('--kernel-cluster-alg', type=str, default=None, help='the algorithm to use for kernel clustering; no need to set up if running univariate GP models')
    parser.add_argument('--flag-plot-kernel-cluster', type=int, default=1, help='the level of visualization; 0: no visualization, 1: visualizing only mode kernels, others: visualizing all')
    parser.add_argument('--opt-config', type=str, required=True, help='the json file that configures optimization setup')

    args = parser.parse_args()

    cv_fold_num = args.cv_fold_num
    cv_seed = args.cv_seed

    mixture_num = args.Q
    rank_num = args.R
    kernel = args.kernel
    eta = args.eta
    beta_lam = args.beta_lam
    prior = args.prior

    exp_prefix = args.exp_prefix
    cohort = args.cohort
    kernel_cluster_alg = args.kernel_cluster_alg

    """
    read config files
    """
    path_dict = json.load(open(args.path_config, 'r'))
    hpc_dict = json.load(open(args.hpc_config, 'r'))
    opt_dict = json.load(open(args.opt_config, 'r'))
    feature_dict = json.load(open(args.feature_config, 'r'))
    feature_name_list = [x['name'] for x in feature_dict['feature_list']]
    feature_id_list = [x['index'] for x in feature_dict['feature_list']]
    feature_num = len(feature_id_list)
    if(rank_num > feature_num):
        print('Warning: rank ({}) is larger than number of outputs ({})'.format(rank_num, feature_num))
    figure_config = args.figure_config
    flag_plot_kc = args.flag_plot_kernel_cluster

    """
    set variables from specified configuration json files
    """
    # setup medgpc path and import modules
    medgpc_path = path_dict['medgpc_path']
    print('medgpc lib path: {}'.format(medgpc_path))
    medgpc_exec_path = os.path.join(medgpc_path, 'medgpc', 'src')

    """
    create experiment dependencies
    """
    kernel_code_dict = {'LMC-SM': 7, 'SE': 0, 'SM': 8}
    try:
        kernel_index = kernel_code_dict[kernel]
    except KeyError:
        print('Undefined kernel type {}'.format(kernel))
        print('Supperted type: ')
        print(kernel_code_dict.keys())
        exit(1)
    except:
        raise NotImplementedError

    prior_code_dict = {'hier-gamma': 2, 'None': 0}
    try:
        prior_index = prior_code_dict[str(prior)]
    except KeyError:
        print('Undefined prior type {}'.format(prior))
        print('Supperted type: ')
        print(prior_code_dict.keys())
        exit(1)
    except:
        raise NotImplementedError

    if(kernel == 'LMC-SM'): # multivariate GP kernels
        if(prior_index == 2): # sparse
            exp_name = '{}_k{}_q{}_r{}_p{}_e{}'.format(exp_prefix, kernel_index, mixture_num, rank_num, prior_index, '{:2.2f}'.format(eta))
        else: # non-sparse
            exp_name = '{}_k{}_q{}_r{}_p{}'.format(exp_prefix, kernel_index, mixture_num, rank_num, prior_index)
    elif((kernel == 'SE') or (kernel == 'SM')): # univariate GP kernels
        if(prior_index != 0):
            print('Warning: prior {} is not applicable to kernel {}; reset to None'.format(prior, kernel))
            prior = 'None'
            prior_index = 0
        if(kernel == 'SM'):
            exp_name = '{}_f{}_k{}_q{}'.format(exp_prefix, feature_id_list[0], kernel_index, mixture_num)
        else:
            exp_name = '{}_f{}_k{}'.format(exp_prefix, feature_id_list[0], kernel_index)
            if(kernel_cluster_alg != 'None'):
                print('Warning: kernel clustering algorithm is unused for kernel {}; reset to None'.format(kernel))
                kernel_cluster_alg = 'None'
    else:
        print('Error: can not recognize kernel {}'.format(kernel))
        exit(1)

    # create directories of this experiment
    print('creating dependencies for experiment named {}'.format(exp_name))
    # setup variables from path config file
    exp_path_dict = {}
    exp_root_path = path_dict['exp_root_path']
    exp_top_dir = os.path.join(exp_root_path, exp_name)
    exp_cfg_dir = os.path.join(exp_top_dir, 'config')
    exp_json_file = os.path.join(exp_cfg_dir, 'exp_setup.json')

    exp_log_dir = os.path.join(exp_top_dir, 'log')
    exp_train_dir = os.path.join(exp_top_dir, 'train')
    exp_test_dir = os.path.join(exp_top_dir, 'test')
    exp_script_dir = os.path.join(exp_top_dir, 'script')
    exp_kernel_dir = os.path.join(exp_top_dir, 'kernel')
    exp_figure_dir = os.path.join(exp_top_dir, 'figure')

    exp_path_dict =  {
                        "exp_top_dir": exp_top_dir,
                        "exp_cfg_dir": exp_cfg_dir,
                        "exp_log_dir": exp_log_dir,
                        "exp_train_dir": exp_train_dir,
                        "exp_test_dir": exp_test_dir,
                        "exp_script_dir": exp_script_dir,
                        "exp_kernel_dir": exp_kernel_dir,
                        "exp_figure_dir": exp_figure_dir,
                        }
    for dir_desc, local_dir in exp_path_dict.items():
        if not os.path.exists(local_dir):
            os.makedirs(local_dir)
        else:
            print('Warning: {} already exists'.format(local_dir))
        os.chmod(local_dir, 0o774)

    # add data path information
    exp_path_dict['data_dir'] = os.path.join(path_dict['data_root_path'], cohort)
    exp_path_dict['cohort_id_list'] = path_dict['cohort_id_list']

    """
    write training config file for medgpc
    """
    hyp_bound_file = 'hyp_bound.txt'
    exp_path_dict['hyp_bound_file'] = hyp_bound_file
    write_medgpc_bound(output_dir=exp_cfg_dir, file_name=hyp_bound_file, feature_num=feature_num,
                       kernel_index=kernel_index, mixture_num=mixture_num, rank_num=rank_num,
                       opt_config=opt_dict)

    # read patient ids
    pid_list_file = os.path.join(exp_path_dict['data_dir'], exp_path_dict['cohort_id_list'])
    print('patient id file: {}'.format(pid_list_file))
    pid_array = np.genfromtxt(pid_list_file, dtype=str)
    print('# number of patients read: {}'.format(len(pid_array)))

    # write assignment of cross-validation
    cv_assign_file = os.path.join(exp_cfg_dir, 'cv_assign.txt')
    np.random.seed(cv_seed)
    cv_assign_array = np.int_(np.mod(np.random.permutation(len(pid_array)), cv_fold_num))
    for cv in range(cv_fold_num):
        print('fold {} has {} patients'.format(cv, len(np.where(cv_assign_array == cv)[0])))
    np.savetxt(cv_assign_file, cv_assign_array, delimiter='\n', fmt='%d')

    # write final json file for the experiment
    write_medgpc_config_json(exp_config_file=exp_json_file, exp_path_config=exp_path_dict,
                            kernel=kernel, kernel_index=kernel_index,
                            feature_list=feature_id_list, prior=prior, prior_index=prior_index, 
                            eta=eta, beta_lam=beta_lam, mixture_num=mixture_num, rank_num=rank_num, 
                            opt_config=opt_dict, cv_fold_num=cv_fold_num, cv_assign_file=cv_assign_file)


    """
    output computing cluster scripts
    """
    # read cluster settings
    for mode in ['train', 'test']:
        # generate customized cluster basic scripts
        template_file = hpc_dict['{}_template'.format(mode)]
        f = open(template_file, 'r')
        template_cmds = f.readlines()
        f.close()
        for dd in hpc_dict['{}_config'.format(mode)]:
            write_scheduler_sh( output_dir=exp_script_dir,
                                output_file_name=dd['script_name'], 
                                scheduler_type=dd['type'],
                                base_cmds=template_cmds, 
                                memory=dd['mem'], 
                                time=dd['runtime'], 
                                thread=dd['thread'], 
                                node_thread=dd['host_thread_limit'],
                                )
        # generate scripts to submit all jobs
        batch_script = os.path.join(exp_top_dir, 'run_{}_all.sh'.format(mode))
        f = open(batch_script, 'w')
        for pidx, curr_pid in enumerate(pid_array):
            # count all the data points for this subject
            curr_psize = get_sample_num(data_dir=path_dict['data_root_path'], cohort=cohort, 
                                        feature_list=feature_id_list, pid=curr_pid)
            if(curr_psize > 6000):
                print('Warning: patient id {} has {} observations'.format(curr_pid, curr_psize))

            # decide which basic script to call
            for dd in hpc_dict['{}_config'.format(mode)]:
                if((curr_psize >= dd['min_mat_size']) and (curr_psize < dd['max_mat_size'])):
                    break

            cmd = ''
            if(dd['type'] == 'slurm'):
                job_name = '{}_{}_{}'.format(mode, curr_pid, exp_name)
                cmd += 'sbatch --job-name={} '.format(job_name)
                cmd += os.path.join(exp_script_dir, dd['script_name'])
                cmd += ' '
                cmd += '--prefix {} --exe {} --cfg {} --pan {} --thread {} --log-path {}'.format(
                        mode, os.path.join(medgpc_exec_path, path_dict['{}_exec'.format(mode)]), 
                        exp_json_file, curr_pid, dd['thread'], exp_log_dir)
                if(mode == 'test'):
                    cmd += ' --fold {}'.format(cv_assign_array[pidx])
                    cmd += ' --kernclust-alg {}'.format(kernel_cluster_alg)
            elif(dd['type'] == 'pbs'):
                job_name = '{}_{}_{}'.format(mode, curr_pid, exp_name)
                cmd += 'qsub -N {} '.format(job_name)
                cmd += '-v '
                cmd += 'prefix="{}",exe="{}",cfg="{}",pan="{}",thread="{}",log_path="{}"'.format(
                        mode, os.path.join(medgpc_exec_path, path_dict['{}_exec'.format(mode)]), 
                        exp_json_file, curr_pid, dd['thread'], exp_log_dir)
                if(mode == 'test'):
                    cmd += ',fold="{}"'.format(cv_assign_array[pidx])
                    cmd += ',kernclust_alg="{}"'.format(kernel_cluster_alg)
                cmd += ' '
                cmd += os.path.join(exp_script_dir, dd['script_name'])
            else:
                print('Warning: scheduler {} is not supported; sequential command is used'.format(dd['type']))
                cmd += os.path.join(exp_script_dir, dd['script_name'])
                cmd += ' '
                cmd += '--prefix {} --exe {} --cfg {} --pan {} --thread {} --log-path {}'.format(
                        mode, os.path.join(medgpc_exec_path, path_dict['{}_exec'.format(mode)]), 
                        exp_json_file, curr_pid, dd['thread'], exp_log_dir)
                if(mode == 'test'):
                    cmd += ' --fold {}'.format(cv_assign_array[pidx])
                    cmd += ' --kernclust-alg {}'.format(kernel_cluster_alg)

            f.write(cmd + '\n')
        f.close()
        st = os.stat(batch_script)
        os.chmod(batch_script, st.st_mode | stat.S_IEXEC)
        print('Info: execute {} to submit {} jobs'.format(batch_script, mode))

    """
    write kernel clustering scripts
    """
    # read bash template first
    template_file = hpc_dict['kernclust_template']
    f = open(template_file, 'r')
    template_cmds = f.readlines()
    f.close()

    # write script for batch jobs
    kc_batch_script = os.path.join(exp_top_dir, 'run_kernclust_all.sh')
    ff = open(kc_batch_script, 'w')

    dd = hpc_dict['kernclust_config']
    # kernclust_script = os.path.join(medgpc_path, 'medgpc', 'clustering', 'run_kernel_clustering.py')
    kernclust_script = '{}.{}.{}'.format('medgpc', 'clustering', 'run_kernel_clustering')

    for fold in range(-1, cv_fold_num):
        kc_cmd = 'python -m {} '.format(kernclust_script)
        kc_cmd += '--exp-config {} '.format(exp_json_file)
        kc_cmd += '--fold {} '.format(fold)
        kc_cmd += '--kernel-cluster-alg {} '.format(kernel_cluster_alg)
        if(figure_config is not None):
            kc_cmd += '--figure-config {} '.format(figure_config)
        kc_cmd += '--figure-plot-mode {} '.format(flag_plot_kc)
        kc_cmd += '> {}'.format(os.path.join(exp_log_dir, 'fold{}_kernel_clustering.log'.format(fold)))
        fold_script_name = 'fold{}_'.format(fold)+dd['script_name']
        write_scheduler_sh( output_dir=exp_script_dir,
                            output_file_name=fold_script_name, 
                            scheduler_type=dd['type'],
                            base_cmds=template_cmds, 
                            memory=dd['mem'], 
                            time=dd['runtime'], 
                            thread=dd['thread'], 
                            node_thread=dd['host_thread_limit'],
                            extra_cmd=[kc_cmd]
                            )
        if(dd['type'] == 'slurm'):
            job_name = 'kernclust_fold{}_{}'.format(fold, exp_name)
            ff.write('sbatch --job-name={} {}\n'.format(job_name, os.path.join(exp_script_dir, fold_script_name)))
        elif(dd['type'] == 'pbs'):
            job_name = 'kernclust_fold{}_{}'.format(fold, exp_name)
            ff.write('qsub -N {} {}\n'.format(job_name, os.path.join(exp_script_dir, fold_script_name)))
        else:
            ff.write('{}\n'.format(os.path.join(exp_script_dir, fold_script_name)))
    ff.close()
    st = os.stat(kc_batch_script)
    os.chmod(kc_batch_script, st.st_mode | stat.S_IEXEC)
    print('Info: execute {} to submit kernel clustering jobs'.format(kc_batch_script))


    """
    write result collection scripts
    """
    # read bash template first
    template_file = hpc_dict['eval_template']
    f = open(template_file, 'r')
    template_cmds = f.readlines()
    f.close()

    # write script for batch jobs
    eval_batch_script = os.path.join(exp_top_dir, 'run_eval_all.sh')
    ff = open(eval_batch_script, 'w')

    dd = hpc_dict['eval_config']
    eval_script = '{}.{}.{}'.format('medgpc', 'evaluation', 'run_medgpc_eval')

    for test_mode in ['mean_wo_update', 'mean_w_update']:
        kc_cmd = 'python -m {} '.format(eval_script)
        kc_cmd += '--exp-config {} '.format(exp_json_file)
        # kc_cmd += '--fold {} '.format(fold)
        kc_cmd += '--test-mode {} '.format(test_mode)
        kc_cmd += '> {}'.format(os.path.join(exp_log_dir, 'eval_{}.log'.format(test_mode)))
        fold_script_name = '{}_'.format(test_mode)+dd['script_name']
        write_scheduler_sh( output_dir=exp_script_dir,
                            output_file_name=fold_script_name, 
                            scheduler_type=dd['type'],
                            base_cmds=template_cmds, 
                            memory=dd['mem'], 
                            time=dd['runtime'], 
                            thread=dd['thread'], 
                            node_thread=dd['host_thread_limit'],
                            extra_cmd=[kc_cmd]
                            )
        if(dd['type'] == 'slurm'):
            job_name = 'eval_{}_{}'.format(test_mode, exp_name)
            ff.write('sbatch --job-name={} {}\n'.format(job_name, os.path.join(exp_script_dir, fold_script_name)))
        elif(dd['type'] == 'pbs'):
            job_name = 'eval_{}_{}'.format(test_mode, exp_name)
            ff.write('qsub -N {} {}\n'.format(job_name, os.path.join(exp_script_dir, fold_script_name)))
        else:
            ff.write('{}\n'.format(os.path.join(exp_script_dir, fold_script_name)))
    ff.close()
    st = os.stat(eval_batch_script)
    os.chmod(eval_batch_script, st.st_mode | stat.S_IEXEC)
    print('Info: execute {} to submit result collection jobs'.format(eval_batch_script))

