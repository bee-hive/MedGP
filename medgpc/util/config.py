import os
import json


def write_medgpc_config_json(exp_config_file, exp_path_config, kernel, kernel_index,
                             feature_list, prior, prior_index, eta, beta_lam, mixture_num, rank_num, 
                             opt_config, cv_fold_num, cv_assign_file):
     exp_setup_dict = exp_path_config.copy()
     exp_setup_dict['kernel'] = kernel
     exp_setup_dict['kernel_index'] = kernel_index

     exp_setup_dict['prior'] = str(prior)
     exp_setup_dict['prior_index'] = prior_index

     exp_setup_dict['Q'] = mixture_num
     exp_setup_dict['D'] = len(feature_list)
     exp_setup_dict['R'] = rank_num
     feature_str = ''
     for ff in feature_list:
          feature_str += '{} '.format(ff)
     exp_setup_dict['feature_index'] = feature_str

     exp_setup_dict['eta'] = eta
     exp_setup_dict['beta_lam'] = beta_lam

     exp_setup_dict['cv_fold_num'] = cv_fold_num
     exp_setup_dict['cv_assign_file'] = cv_assign_file

     exp_setup_dict['random_init_num'] = opt_config['random_init_num']
     exp_setup_dict['random_seed'] = opt_config['random_seed']
     exp_setup_dict['top_iteration_num'] = opt_config['top_iteration_num']
     exp_setup_dict['iteration_num_per_update'] = opt_config['iteration_num_per_update']
     exp_setup_dict['online_learn_rate'] = opt_config['online_learn_rate']
     exp_setup_dict['online_momentum'] = opt_config['online_momentum']
     json.dump(exp_setup_dict, open(exp_config_file, 'w'), indent=4)


def write_medgpc_bound(output_dir, file_name, feature_num, kernel_index, mixture_num, rank_num, opt_config):
     # output hyperparameter file
     f = open(os.path.join(output_dir, file_name), 'w')
     if(kernel_index == 7):
          # output upper/lower bound of Gaussian noise parameter
          for i in range(0, feature_num):
               f.write('{:6.6f}\n'.format(opt_config['lower_bound_noise']))
               f.write('{:6.6f}\n'.format(opt_config['upper_bound_noise']))

          # output upper/lower bound of elements in A matrix
          for i in range(0, mixture_num*feature_num*rank_num):
               f.write('{:6.6f}\n'.format(opt_config['lower_bound_a']))
               f.write('{:6.6f}\n'.format(opt_config['upper_bound_a']))

          # output upper/lower bound of period parameter
          for i in range(0, mixture_num):
               f.write('{:6.6f}\n'.format(opt_config['lower_bound_period']))
               f.write('{:6.6f}\n'.format(opt_config['upper_bound_period']))

          # output upper/lower bound of lengthscale parameter
          for i in range(0, mixture_num):
               f.write('{:6.6f}\n'.format(opt_config['lower_bound_lengthscale']))
               f.write('{:6.6f}\n'.format(opt_config['upper_bound_lengthscale']))

          # output upper/lower bound of lambda parameter
          for i in range(0, mixture_num*feature_num):
               f.write('{:6.6f}\n'.format(opt_config['lower_bound_lambda']))
               f.write('{:6.6f}\n'.format(opt_config['upper_bound_lambda']))

     elif(kernel_index == 0):
          # output upper/lower bound of likelihood parameter
          f.write('{:6.6f}\n'.format(opt_config['lower_bound_noise']))
          f.write('{:6.6f}\n'.format(opt_config['upper_bound_noise']))

          # output upper/lower bound of lengthscale parameter
          f.write('{:6.6f}\n'.format(opt_config['lower_bound_lengthscale']))
          f.write('{:6.6f}\n'.format(opt_config['upper_bound_lengthscale']))

          # output upper/lower bound of scalefactor parameter
          assert opt_config['lower_bound_scale'] > 0
          f.write('{:6.6f}\n'.format(opt_config['lower_bound_scale']))
          f.write('{:6.6f}\n'.format(opt_config['upper_bound_scale']))

     elif(kernel_index == 8):
          # output upper/lower bound of likelihood parameter
          f.write('{:6.6f}\n'.format(opt_config['lower_bound_noise']))
          f.write('{:6.6f}\n'.format(opt_config['upper_bound_noise']))

          # output upper/lower bound of scalefactor parameter
          for i in range(0, mixture_num):
               assert opt_config['lower_bound_scale'] > 0
               f.write('{:6.6f}\n'.format(opt_config['lower_bound_scale']))
               f.write('{:6.6f}\n'.format(opt_config['upper_bound_scale']))

          # output upper/lower bound of mu parameter
          for i in range(0, mixture_num):
               f.write('{:6.6f}\n'.format(opt_config['lower_bound_period']))
               f.write('{:6.6f}\n'.format(opt_config['upper_bound_period']))

          # output upper/lower bound of v parameter
          for i in range(0, mixture_num):
               f.write('{:6.6f}\n'.format(opt_config['lower_bound_lengthscale']))
               f.write('{:6.6f}\n'.format(opt_config['upper_bound_lengthscale']))

     else:
          print('Erorr: input kernel index {} not supported'.format(kernel_index))
          raise NotImplementedError
     f.close()

