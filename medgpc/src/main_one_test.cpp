/*
-------------------------------------------------------------------------
This is the top file for online imputation testing for one patient.
-------------------------------------------------------------------------
*/
#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <string.h>
#include <limits>
#include <algorithm>

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mkl.h>
#include <omp.h>
#include <time.h>

#include "dataio/c_experiment.h"
#include "util/c_objective.h"
#include "util/c_objective_one.h"
#include "util/global_settings.h"
#include "core/c_hyperparam.h"
#include "core/gp_model_include.h"
#include "core/gp_regression.h"

#define CMD_NUM 11

using namespace std;

void run_model_SE(c_experiment &curr_exp, string PAN, int thread_num, int fold, const string &kernel_clust_alg);
void run_model_SM(c_experiment &curr_exp, string PAN, int thread_num, int fold, const string &kernel_clust_alg);
void run_model_LMC_SM(c_experiment &curr_exp, string PAN, int thread_num, int fold, const string &kernel_clust_alg);
void run_test_one(  c_experiment &curr_exp, string PAN, int thread_num, int fold,
                    bool flag_update, string &output_prefix, const string &kernel_clust_alg,
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
        cout << "\t --fold:\t the integer to indicate the cross-validation fold" << endl;
        cout << "\t --kernclust-alg:\t the algorithm name used for kernel clustering" << endl;

        exit(1);
    }
    string exp_cfg;
    int thread_num, patient_fold;
    string patient_PAN;
    string kernel_clust_alg = "None";
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
        else if(!strcmp(argv[i], "--fold")){
            patient_fold = atoi(argv[++i]);
            assert(patient_fold >= 0);
        }
        else if(!strcmp(argv[i], "--kernclust-alg")){
            kernel_clust_alg = argv[++i];
        }
        else{
            cout << "Error: unknown argument: " << argv[i] << endl;
            exit(1);
        }
    }
    cout << "current configuration file: " << exp_cfg << endl;
    cout << "current training patient: " << patient_PAN << endl;
    cout << "current fold index for PAN " << patient_PAN << " = " << patient_fold << endl;
    cout << "current threading number for matrix operation: " << thread_num << endl;
    cout << "kernel clustering algorithm: " << kernel_clust_alg << endl;

    // threading setup
    omp_set_nested(1);

    // new experiments
    c_experiment curr_exp(exp_cfg);

    // run the training routine
    time_t t1, t2;
    int kernel_setup = curr_exp.get_kernel_index();
    time(&t1);
    if(kernel_setup == 0){
        run_model_SE(curr_exp, patient_PAN, thread_num, patient_fold, kernel_clust_alg);
    }
    else if(kernel_setup == 8){
        run_model_SM(curr_exp, patient_PAN, thread_num, patient_fold, kernel_clust_alg);
    }
    else if(kernel_setup == 7){
        run_model_LMC_SM(curr_exp, patient_PAN, thread_num, patient_fold, kernel_clust_alg);
    }
    else{
        cout << "Error: not supported kernel type " << kernel_setup << endl;
        exit(1);
    }
    time(&t2);
    cout << "Finish all jobs. Total elapsed time = " << difftime(t2, t1) << " seconds" << endl;
    return 0;
}

void run_model_LMC_SM(c_experiment &curr_exp, string PAN, int thread_num, int fold, const string &kernel_clust_alg){
    vector<int>     test_kernel_param;
    test_kernel_param = curr_exp.get_test_kernel_param(fold, kernel_clust_alg);
    cout << "# of mixture for testing: " << test_kernel_param[0] << endl;

    c_kernel_LMC_SM             kernel(test_kernel_param);
    c_inference_prior           inffunc(thread_num);
    c_meanfunc_zero             meanfunc;
    c_likelihood_gaussianMO     likfunc(curr_exp.get_lik_param());
    c_prior                     prior(curr_exp.get_test_cov_num(fold, kernel_clust_alg),
                                      curr_exp.get_mean_num(),
                                      curr_exp.get_lik_num());
    c_kernel        *kptr = &kernel;
    c_meanfunc      *mptr = &meanfunc;
    c_likelihood    *lptr = &likfunc;
    c_inference     *iptr = &inffunc;
    c_prior         *pptr = &prior;

    string prefix_no_update, prefix_w_update;
    prefix_no_update = "mean_wo_update";
    prefix_w_update = "mean_w_update";
    run_test_one(curr_exp, PAN, thread_num, fold, false, prefix_no_update, kernel_clust_alg, kptr, mptr, lptr, iptr, pptr);
    run_test_one(curr_exp, PAN, thread_num, fold, true, prefix_w_update, kernel_clust_alg, kptr, mptr, lptr, iptr, pptr);
}

void run_model_SE(c_experiment &curr_exp, string PAN, int thread_num, int fold, const string &kernel_clust_alg){
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

    string prefix_no_update, prefix_w_update;
    prefix_no_update = "mean_wo_update";
    prefix_w_update = "mean_w_update";
    run_test_one(curr_exp, PAN, thread_num, fold, false, prefix_no_update, kernel_clust_alg, kptr, mptr, lptr, iptr, pptr);
    run_test_one(curr_exp, PAN, thread_num, fold, true, prefix_w_update, kernel_clust_alg, kptr, mptr, lptr, iptr, pptr);
}

void run_model_SM(c_experiment &curr_exp, string PAN, int thread_num, int fold, const string &kernel_clust_alg){
    vector<int>     test_kernel_param;
    test_kernel_param = curr_exp.get_test_kernel_param(fold, kernel_clust_alg);
    cout << "# of mixture for testing: " << test_kernel_param[0] << endl;

    c_kernel_SM                 kernel(test_kernel_param);
    c_inference_exact           inffunc(thread_num);
    c_meanfunc_zero             meanfunc;
    c_likelihood_gaussian       likfunc;
    c_prior                     prior(curr_exp.get_test_cov_num(fold, kernel_clust_alg),
                                      curr_exp.get_mean_num(),
                                      curr_exp.get_lik_num());
    c_kernel        *kptr = &kernel;
    c_meanfunc      *mptr = &meanfunc;
    c_likelihood    *lptr = &likfunc;
    c_inference     *iptr = &inffunc;
    c_prior         *pptr = &prior;

    string prefix_no_update, prefix_w_update;
    prefix_no_update = "mean_wo_update";
    prefix_w_update = "mean_w_update";
    run_test_one(curr_exp, PAN, thread_num, fold, false, prefix_no_update, kernel_clust_alg, kptr, mptr, lptr, iptr, pptr);
    run_test_one(curr_exp, PAN, thread_num, fold, true, prefix_w_update, kernel_clust_alg, kptr, mptr, lptr, iptr, pptr);
}

void run_test_one(  c_experiment &curr_exp, string PAN, int thread_num, int fold,
                    bool flag_update, string &output_prefix, const string &kernel_clust_alg,
                    c_kernel       *&kptr, 
                    c_meanfunc     *&mptr, 
                    c_likelihood   *&lptr, 
                    c_inference    *&iptr,
                    c_prior        *&pptr){
    cout << "running online imputation: ";
    if(flag_update){
        cout << "with online updating" << endl;
    }
    else{
        cout << "without online updating" << endl;
    }
    cout << "testing patinet: " << PAN << " in cross-validation fold " << fold << endl;

    // pre-load the data
    vector<int>     test_kernel_param;
    test_kernel_param = curr_exp.get_test_kernel_param(fold, kernel_clust_alg);
    cout << "test_kernel_param: " << test_kernel_param[0] << endl;

    time_t t1, t2, t3, t4, t5, t6;
    vector<int> meta_array;
    vector<float> time_array;
    vector<float> value_array;
    curr_exp.get_one_patient_data(PAN, meta_array, time_array, value_array);
    cout << "number of data points = " << time_array.size() << endl;

    bool test_flag = true;
    time(&t1);
    if(int(time_array.size()) == 0){
        cout << "Warning: no samples for testing" << endl;
        test_flag = false;
    }
    else{
        // sort and unique current time points
        vector<float>::iterator it;
        vector<float> unique_time_array(time_array);
        sort(unique_time_array.begin(), unique_time_array.end());
        it = unique(unique_time_array.begin(), unique_time_array.end());
        unique_time_array.resize( distance(unique_time_array.begin(),it) );

        // find out the earliest time point
        cout << "local min. time stamp = " << unique_time_array[0] << endl;
        cout << "total # of time stamps: " << time_array.size() << endl;
        cout << "total # of unique time stamps: " << unique_time_array.size() << endl;

        // setup testing thread # for kernel
        kptr -> set_kernel_grad_thread(thread_num);
        cout << "number of threads for gradient computation = " 
             << kptr -> get_kernel_grad_thread() << endl;

        // setup online updating parameters
        double learn_rate = curr_exp.get_online_learn_rate();
        double momentum = curr_exp.get_online_momentum();
        if(flag_update){
            cout << "online updating learning rate/momentum: " 
                 << learn_rate << "\t" << momentum << endl;
        }

        // get mode parameters estimated in the kernel clustering step
        vector<double> mode_parameter;
        mode_parameter = curr_exp.get_test_mode_param(fold, kernel_clust_alg);
        vector<double> best_parameter(mode_parameter);
        vector<double> delta_parameter(mode_parameter.size(), 0.0);

        // initialize output vectors
        vector<int>     out_impute_feature_idx;
        vector<int>     out_impute_ci_flag;
        vector<double>  out_impute_elapsed_time;
        vector<double>  out_impute_gp_error;
        vector<double>  out_impute_gp_pred;

        // check prior status
        pptr -> init_test_prior(curr_exp.get_kernel_index(), test_kernel_param, mode_parameter);
        // pptr -> print_status();

        // start imputation testing
        float last_update_time = unique_time_array[0];
        for(int tt = 0; tt < int(unique_time_array.size()); tt++){
            if( (tt % 100) == 0 ){
                cout << "finish testing " << tt << "/" 
                     << int(unique_time_array.size()) << " time stamps" << endl;
            }

            // initialize variables for this loop only
            vector<int> past_obs_meta;
            vector<float> past_obs_time;
            vector<float> past_obs_value;

            vector<int> curr_obs_meta;
            vector<float> curr_obs_time;
            vector<float> curr_obs_value;

            // get training observations (all observations before this time point)
            // and testing observations (observations at this time point)
            for(int ii = 0; ii < int(time_array.size()); ii++){
                if(time_array[ii] < unique_time_array[tt]){                    
                    if(flag_update){
                        if(fabs(time_array[ii] - unique_time_array[tt]) <= 72.0){
                            past_obs_meta.push_back(meta_array[ii]);
                            past_obs_time.push_back(time_array[ii]);
                            past_obs_value.push_back(value_array[ii]);
                        }
                    }
                    else{
                        past_obs_meta.push_back(meta_array[ii]);
                        past_obs_time.push_back(time_array[ii]);
                        past_obs_value.push_back(value_array[ii]);
                    }
                }
                else if(time_array[ii] == unique_time_array[tt]){
                    curr_obs_meta.push_back(meta_array[ii]);
                    curr_obs_time.push_back(time_array[ii]);
                    curr_obs_value.push_back(value_array[ii]);
                }
            }

            if(flag_update){
                if((tt > 3) && (unique_time_array[tt]-last_update_time) > 5.0/60.0){
                    last_update_time = unique_time_array[tt];
                    c_objective_one     curr_objfunc(
                                                    curr_exp.get_kernel_index(),
                                                    test_kernel_param,
                                                    past_obs_meta, 
                                                    past_obs_time,
                                                    past_obs_value
                                                    );
                    c_objective  *obj_ptr = &curr_objfunc;

                    double best_loss;
                    vector<double> best_grads;
                    bool obj_flag;
                    obj_flag = obj_ptr -> compute_objective(
                                                        true, best_parameter, 
                                                        best_loss, best_grads, 
                                                        kptr, mptr, lptr, iptr, pptr);
                    if(obj_flag){
                        for(int h = 0; h < int(mode_parameter.size()); h++){
                            bool prior_flag = pptr -> get_one_prior_flag(h);
                            int prior_type = pptr -> get_one_prior_type(h);
                            if( (!prior_flag) | (prior_type != 0) ){
                                delta_parameter[h] = momentum*delta_parameter[h] + learn_rate*best_grads[h];
                                best_parameter[h] -= delta_parameter[h];
                            }
                            // else{
                            //     cout << "no update for mode parameter[" << h << "] = " << mode_parameter[h];
                            // }
                        }
                    }
                    else{
                        cout << "Warning: failed to update at t[" << tt << "] = " << unique_time_array[tt]
                             << "; reset to mode parameters" << endl;
                        best_parameter = mode_parameter;
                        for(int h = 0; h < int(mode_parameter.size()); h++){
                            delta_parameter[h] = 0.0;
                        }
                    }
                } // if((unique_time_array[tt]-unique_time_array[0]) >= 72.0)
            } // if(flag_update)

            // do imputation for each testing observation:
            for(int jj = 0; jj < int(curr_obs_time.size()); jj++){
                // update training data
                vector<int>     impute_train_meta(past_obs_meta); 
                vector<float>   impute_train_time(past_obs_time);
                vector<float>   impute_train_value(past_obs_value);

                // if there are observations from different covariates measured at the same point
                // include them as well
                for(int kk = 0; kk < int(curr_obs_meta.size()); kk++){
                    if(kk != jj){ // not including the current testing point
                        impute_train_meta.push_back(curr_obs_meta[kk]);
                        impute_train_time.push_back(curr_obs_time[kk]);
                        impute_train_value.push_back(curr_obs_value[kk]);
                    }
                }

                // create vector of testing
                vector<int>     impute_test_meta; 
                vector<float>   impute_test_time;
                impute_test_meta.push_back(curr_obs_meta[jj]);
                impute_test_time.push_back(curr_obs_time[jj]);

                // retrain the model for new cholesky factor
                if(int(impute_train_time.size()) > 0){
                    c_hyperparam    hyp(
                                        best_parameter, 
                                        kptr -> get_kernel_hyp_num(), 
                                        mptr -> get_meanfunc_hyp_num(), 
                                        lptr -> get_likfunc_hyp_num()
                                        );
                    kptr -> set_kernel_hyp(hyp.get_hyp_cov());
                    mptr -> set_meanfunc_hyp(hyp.get_hyp_mean());
                    lptr -> set_likfunc_hyp(hyp.get_hyp_lik());

                    GP_Regression gpr_model(1, kptr, mptr, lptr, iptr, pptr);
                    gpr_model.train(false, impute_train_meta, impute_train_time, impute_train_value);
                    
                    bool curr_chol_flag;
                    curr_chol_flag = gpr_model.get_flag_trained();
                    if(curr_chol_flag){
                        vector< vector<float> > impute_one_posterior = 
                                                gpr_model.predict(
                                                                    impute_train_meta, 
                                                                    impute_test_meta, 
                                                                    impute_train_time, 
                                                                    impute_train_value, 
                                                                    impute_test_time
                                                                    );
                        out_impute_gp_pred.push_back((double)impute_one_posterior[0][0]);
                        double impute_error = impute_one_posterior[0][0] - curr_obs_value[jj];
                        out_impute_gp_error.push_back(impute_error);

                        if(fabs(impute_error) <= 1.96*sqrt(impute_one_posterior[1][0])){
                            out_impute_ci_flag.push_back(1);
                        }
                        else{
                            out_impute_ci_flag.push_back(0);
                        }
                    }
                    else{
                        cout << "Warning: failed to predict with current parameter" << endl;
                        out_impute_gp_pred.push_back(0.0);
                        double impute_error = 0.0 - curr_obs_value[jj];
                        out_impute_gp_error.push_back(impute_error);

                        double prior_var = exp(mode_parameter[impute_test_meta[0]]);
                        if(fabs(impute_error) <= 1.96*prior_var){
                            out_impute_ci_flag.push_back(1);
                        }
                        else{
                            out_impute_ci_flag.push_back(0);
                        }
                    }
                } // if(int(impute_train_time.size()) > 0)
                else{
                    cout << "Warning: no training observations; predict with zero mean" << endl;
                    out_impute_gp_pred.push_back(0.0);
                    double impute_error = 0.0 - curr_obs_value[jj];
                    out_impute_gp_error.push_back(impute_error);

                    double prior_var = exp(mode_parameter[impute_test_meta[0]]);
                    if(fabs(impute_error) <= 1.96*prior_var){
                        out_impute_ci_flag.push_back(1);
                    }
                    else{
                        out_impute_ci_flag.push_back(0);
                    }
                } // 
                int curr_test_fidx = curr_exp.get_feature_index()[impute_test_meta[0]];
                out_impute_feature_idx.push_back(curr_test_fidx);
                out_impute_elapsed_time.push_back(impute_test_time[0]-unique_time_array[tt]);
            } // for(int jj = 0; jj < int(curr_obs_time.size()); jj++)
        } // for(int tt = 0; tt < int(unique_time_array.size()); tt++)

        // output imputation results
        if(int(out_impute_gp_pred.size()) > 0){
            string file_prefix = curr_exp.get_exp_test_dir() + "test_" + output_prefix + "_";
            string file_name;
            
            file_name = file_prefix + "feature_" + PAN;
            curr_exp.output_int_txt(file_name, out_impute_feature_idx);

            file_name = file_prefix + "etime_" + PAN;
            curr_exp.output_double_bin(file_name, out_impute_elapsed_time);

            file_name = file_prefix + "ci_" + PAN;
            curr_exp.output_int_txt(file_name, out_impute_ci_flag);

            file_name = file_prefix + "error_" + PAN;
            curr_exp.output_double_bin(file_name, out_impute_gp_error);

            file_name = file_prefix + "pred_" + PAN;
            curr_exp.output_double_bin(file_name, out_impute_gp_pred);
        }
    }
    string file_name;
    file_name = curr_exp.get_exp_test_dir() + "test_" + output_prefix + "_"
                + "flag_" + PAN;
    vector<int> flag_vec;
    flag_vec.push_back((int)test_flag);
    curr_exp.output_int_txt(file_name, flag_vec);

    time(&t2);
    cout << "finish (" <<  output_prefix << ") testing individual PAN " << PAN 
    << " w/ " << int(time_array.size())<< " samples" 
    << "; flag = " << test_flag
    << "; elapsed time = " << difftime(t2, t1) << " seconds" << endl;

}

