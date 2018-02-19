/*
-------------------------------------------------------------------------
This is the top file for training a GP model for one patient.
-------------------------------------------------------------------------
*/
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <string.h>
#include <limits>

#include <stdio.h>
#include <stdlib.h>
#include <mkl.h>
#include <omp.h>
#include <time.h>

#include "dataio/c_experiment.h"
#include "util/c_objective.h"
#include "util/c_objective_one.h"
#include "util/c_optimizer_scg.h"
#include "util/c_optimizer_varEM.h"
#include "util/global_settings.h"
#include "core/gp_model_include.h"

#define CMD_NUM 7

using namespace std;

void run_model_SE(c_experiment &curr_exp, string PAN, int thread_num);
void run_model_SM(c_experiment &curr_exp, string PAN, int thread_num);
void run_model_LMC_SM(c_experiment &curr_exp, string PAN, int thread_num);
void run_train_one( c_experiment &curr_exp, string PAN, int thread_num,
                    c_kernel       *&kptr, 
                    c_meanfunc     *&mptr, 
                    c_likelihood   *&lptr, 
                    c_inference    *&iptr,
                    c_prior        *&pptr);

int main(int argc, const char* argv[]){
    // read in current experiment & hyperparameter settings
    if(argc != CMD_NUM){
        cout << "ERROR: incorrect number of argument received!" << endl;
        cout << "expect " << CMD_NUM << " but received " << argc << endl;
        cout << "usage:" << endl;
        cout << "\t --cfg:\t the JSON configuration file" << endl;
        cout << "\t --pan:\t ID of the training patient" << endl;
        cout << "\t --thread:\t number of threads for LAPACK/BLAS matrix operation" << endl;

        exit(1);
    }
    string exp_cfg;
    string patient_PAN;
    int thread_num;
    for(int i = 1; i < argc; i++){
        if(!strcmp(argv[i], "--cfg")){
            exp_cfg = argv[++i];
        }
        else if(!strcmp(argv[i], "--pan")){
            patient_PAN = argv[++i];
        }
        else if(!strcmp(argv[i], "--thread")){
            thread_num = atoi(argv[++i]);
        }
        else{
            cout << "Error: unknown argument: " << argv[i] << endl;
            exit(1);
        }
    }
    cout << "current configuration file: " << exp_cfg << endl;
    cout << "current training patient: " << patient_PAN << endl;
    cout << "current threading number for matrix operation: " << thread_num << endl;

    // threading setup
    omp_set_nested(1);

    // new experiments
    c_experiment curr_exp(exp_cfg);

    // run the training routine
    time_t t1, t2;
    int kernel_setup = curr_exp.get_kernel_index();
    time(&t1);
    if(kernel_setup == 0){
        run_model_SE(curr_exp, patient_PAN, thread_num);
    }
    else if(kernel_setup == 8){
        run_model_SM(curr_exp, patient_PAN, thread_num);
    }
    else if(kernel_setup == 7){
        run_model_LMC_SM(curr_exp, patient_PAN, thread_num);
    }
    else{
        cout << "Error: not supported kernel type " << kernel_setup << endl;
        exit(1);
    }
    time(&t2);
    cout << "Finish all jobs. Total elapsed time = " << difftime(t2, t1) << " seconds" << endl;
    return 0;
}

void run_model_LMC_SM(c_experiment &curr_exp, string PAN, int thread_num){
    c_kernel_LMC_SM             kernel(curr_exp.get_kernel_param());
    c_inference_prior           inffunc(thread_num);
    c_meanfunc_zero             meanfunc;
    c_likelihood_gaussianMO     likfunc(curr_exp.get_lik_param());
    c_prior                     prior(curr_exp.get_cov_num(),
                                      curr_exp.get_mean_num(),
                                      curr_exp.get_lik_num());
    c_kernel        *kptr = &kernel;
    c_meanfunc      *mptr = &meanfunc;
    c_likelihood    *lptr = &likfunc;
    c_inference     *iptr = &inffunc;
    c_prior         *pptr = &prior;

    run_train_one(curr_exp, PAN, thread_num, kptr, mptr, lptr, iptr, pptr);
}

void run_model_SE(c_experiment &curr_exp, string PAN, int thread_num){
    c_kernel_SE                 kernel(curr_exp.get_kernel_param());
    c_inference_exact           inffunc(thread_num);            
    c_meanfunc_zero             meanfunc;
    c_likelihood_gaussian       likfunc;
    c_prior                     prior(curr_exp.get_cov_num(),
                                      curr_exp.get_mean_num(),
                                      curr_exp.get_lik_num());
    c_kernel        *kptr = &kernel;
    c_meanfunc      *mptr = &meanfunc;
    c_likelihood    *lptr = &likfunc;
    c_inference     *iptr = &inffunc;
    c_prior         *pptr = &prior;

    run_train_one(curr_exp, PAN, thread_num, kptr, mptr, lptr, iptr, pptr);
}

void run_model_SM(c_experiment &curr_exp, string PAN, int thread_num){
    c_kernel_SM                 kernel(curr_exp.get_kernel_param());
    c_inference_exact           inffunc(thread_num);            
    c_meanfunc_zero             meanfunc;
    c_likelihood_gaussian       likfunc;
    c_prior                     prior(curr_exp.get_cov_num(),
                                      curr_exp.get_mean_num(),
                                      curr_exp.get_lik_num());
    c_kernel        *kptr = &kernel;
    c_meanfunc      *mptr = &meanfunc;
    c_likelihood    *lptr = &likfunc;
    c_inference     *iptr = &inffunc;
    c_prior         *pptr = &prior;

    run_train_one(curr_exp, PAN, thread_num, kptr, mptr, lptr, iptr, pptr);
}

void run_train_one( c_experiment &curr_exp, string PAN, int thread_num,
                    c_kernel       *&kptr, 
                    c_meanfunc     *&mptr, 
                    c_likelihood   *&lptr, 
                    c_inference    *&iptr,
                    c_prior        *&pptr){
    // model setup
    iptr -> print_inffunc();
    kptr -> print_kernel();

    // check input train data is valid
    cout << "running individual training..." << endl;
    cout << "current patinet PAN = " << PAN << endl;

    // get the initialization hypers for one selected patient
    vector< vector<double> > global_hyp_array;
    curr_exp.get_global_hyp(global_hyp_array);

    time_t t1, t2, t3, t4, t5, t6;

    // pre-load the data
    vector<int> meta_array;
    vector<float> time_array;
    vector<float> value_array;
    curr_exp.get_one_patient_data(PAN, meta_array, time_array, value_array);
    cout << "current number of data points = " << time_array.size() << endl;

    // check the number of observations for each output
    time(&t1);
    bool flag_data = false;
    bool sample_flag = true;
    vector<int> count_array;
    for(int f = 0; f < int(curr_exp.get_feature_index().size()); f++){
        count_array.push_back(0);
    }
    for(int t = 0; t < int(time_array.size()); t++){
        count_array[meta_array[t]] += 1;
    }
    for(int f = 0; f < int(count_array.size()); f++){
        if(count_array[f] < 2){
            sample_flag = false;
            break;
        }
    }

    // model setup
    if(!sample_flag){
        cout << "skip due to insufficient # of samples" << endl;
        flag_data = false;
    }
    else{
        // setup objective functions
        c_objective_one     curr_objfunc(
                                        curr_exp.get_kernel_index(),
                                        curr_exp.get_kernel_param(),
                                        meta_array, 
                                        time_array,
                                        value_array
                                        );
        c_objective  *obj_ptr = &curr_objfunc;

        double best_loss = numeric_limits<double>::max();
        vector<double> best_init;
        vector<double> opt_parameter;
        bool success;

        // initializa parameters
        pptr -> initialize_param(
                                curr_exp.get_cov_num(), 
                                curr_exp.get_mean_num(), 
                                curr_exp.get_lik_num()
                                );
        cout << "finish initialization of prior" << endl;
        time(&t3);
        for(int init = 0; init < curr_exp.get_scg_init_num(); init++){
            time(&t5);
            double curr_loss;
            vector<double> restart_hyp;
            success = curr_objfunc.compute_objective(
                                                    false, 
                                                    global_hyp_array[init], 
                                                    curr_loss, 
                                                    restart_hyp,
                                                    kptr, mptr, lptr, iptr, pptr
                                                    );
            if(success){
                if(curr_loss < best_loss){
                    best_loss = curr_loss;
                    best_init = global_hyp_array[init];
                }
            }
            else{
                cout << "WARNING: failed in computing objective!" << endl;
                break;
            }
            // debugging output
            if(init % 100 == 0){
                cout << "finish init. " << init << endl;
            }
        }
        time(&t4);
        cout << "INFO: finish initialization " << curr_exp.get_scg_init_num()
             << "; time usage = " << difftime(t4, t3) << " seconds" << endl;
        string file_name = curr_exp.get_exp_train_dir() + "train_init_hyp_" + PAN;
        curr_exp.output_double_bin(file_name, best_init);

        // start optimization
        if(success){
            pptr -> setup_param(
                                curr_exp.get_kernel_index(), 
                                curr_exp.get_kernel_param(), 
                                curr_exp.get_prior_mode(),
                                curr_exp.get_prior_hyp()
                                );
            time(&t5);
            cout << "start doing optimization" << endl;
            if(curr_exp.get_prior_mode() == 2){ // sprase hier. gamma prior
                c_optimizer_varEM     curr_optfunc;
                curr_optfunc.set_sub_opt_iter(curr_exp.get_prior_sub_opt_iter());
                curr_optfunc.optimize(  (-1)*curr_exp.get_scg_max_iter_num(),
                                        best_init,
                                        obj_ptr,
                                        true,
                                        best_loss, 
                                        opt_parameter,
                                        kptr, mptr, lptr, iptr, pptr
                                        );
            }
            else{ // no regularization
                c_optimizer_scg curr_optfunc;
                curr_optfunc.optimize(  (-1)*curr_exp.get_scg_max_iter_num(),
                                        best_init,
                                        obj_ptr,
                                        false,
                                        best_loss, 
                                        opt_parameter,
                                        kptr, mptr, lptr, iptr, pptr
                                        );
            }
            time(&t6);
            cout << "total time for doing optimization for patient " << PAN
                     << " = " << difftime(t6, t5) << " seconds" << endl;    

            // output trained hyperparameter results
            string file_name = curr_exp.get_exp_train_dir() + "train_hyp_" + PAN;
            curr_exp.output_double_bin(file_name, opt_parameter);

            if(curr_exp.get_prior_mode() == 2){
                file_name = curr_exp.get_exp_train_dir() + "train_var_hyp_" + PAN;
                curr_exp.output_double_bin(file_name, pptr -> get_cov_varEM_all()); 
            }
        } // successful initialization
        
        flag_data = success;
        time(&t2);
        cout << "finish individual id: " << PAN << " w/ " 
        << int(time_array.size())<< " samples" 
        << "; flag = " << flag_data
        << "; elapsed time = " << difftime(t2, t1) << " seconds" << endl;
    } // pass input data quality check
    
    string file_name;
    vector<int> flag_vec(1, 0);
    file_name = curr_exp.get_exp_train_dir() + "train_num_" + PAN;
    flag_vec[0] = int(time_array.size());
    curr_exp.output_int_txt(file_name, flag_vec);
    
    file_name = curr_exp.get_exp_train_dir() + "train_flag_" + PAN;
    flag_vec[0] = ((int) flag_data);
    curr_exp.output_int_txt(file_name, flag_vec);

}