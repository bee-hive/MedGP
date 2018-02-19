/*
-------------------------------------------------------------------------
This is the main file for Gaussian process regression.
Need to include header file: gp_regression.h
-------------------------------------------------------------------------
*/
#include <iostream>
#include <vector>
#include <assert.h>
#include <mkl.h>
#include <omp.h>
#include <time.h>
#include "core/gp_model_include.h"
#include "core/gp_regression.h"

using namespace std;

GP_Regression::GP_Regression(){
    dim = 1;
    flag_trained = false;
    inf_thread_num = 1;
    nlm_likelihood = 0.0;
    beta = 0.0;
}
GP_Regression::GP_Regression(
                                const int &input_dim, 
                                c_kernel        *input_kernel, 
                                c_meanfunc      *input_meanfunc, 
                                c_likelihood    *input_likfunc, 
                                c_inference     *input_inffunc,
                                c_prior         *input_prior
                                ){
    dim = input_dim;
    flag_trained = false;
    set_kernel(input_kernel);
    set_meanfunc(input_meanfunc);
    set_likelihood(input_likfunc);
    set_inference(input_inffunc);
    set_prior(input_prior);
    nlm_likelihood = 0.0;
    beta = 0.0;
}
GP_Regression::~GP_Regression(){
    delete[]    chol_alpha;
    delete[]    chol_factor_inv;
}
void GP_Regression::reset(
                            const int &input_dim, 
                            c_kernel        *input_kernel, 
                            c_meanfunc      *input_meanfunc, 
                            c_likelihood    *input_likfunc, 
                            c_inference     *input_inffunc,
                            c_prior         *input_prior
                            ){
    dim = input_dim;
    flag_trained = false;
    set_kernel(input_kernel);
    set_meanfunc(input_meanfunc);
    set_likelihood(input_likfunc);
    set_inference(input_inffunc);
    set_prior(input_prior);
    nlm_likelihood = 0.0;
    beta = 0.0;
}
void GP_Regression::set_kernel(c_kernel *input_kernel){
    kernel = input_kernel;
}
void GP_Regression::set_meanfunc(c_meanfunc *input_meanfunc){
    meanfunc = input_meanfunc;
}
void GP_Regression::set_likelihood(c_likelihood *input_likfunc){
    likfunc = input_likfunc;
}
void GP_Regression::set_inference(c_inference *input_inffunc){
    inffunc = input_inffunc;
    inf_thread_num = inffunc -> get_thread_num();
}
void GP_Regression::set_prior(c_prior *input_prior){
    prior = input_prior;
}

int GP_Regression::get_dim(){
    return dim;
}
bool GP_Regression::get_flag_trained(){
    return flag_trained;
}
double GP_Regression::get_neg_log_mlikelihood(){
    return nlm_likelihood;
}
vector<double> GP_Regression::get_dneg_log_mlikelihood(){
    vector<double> dnlm_likelihood_copy(dnlm_likelihood);
    return dnlm_likelihood_copy;
}
void GP_Regression::set_neg_log_mlikelihood(const double &input_nlml){
    nlm_likelihood = input_nlml;
}
void GP_Regression::set_dneg_log_mlikelihood(const vector<double> &input_dnlml){
    dnlm_likelihood = input_dnlml;
}

void GP_Regression::train(
                            const bool &flag_grad, 
                            const vector<int> &meta, 
                            const vector<float> &x, 
                            const vector<float> &y
                            ){
    int n_train_sample = int(y.size());
    int i, j;

    if(flag_trained){
        delete[] chol_factor_inv;
        delete[] chol_alpha;
        flag_trained = false;
    }
    chol_factor_inv = new float[n_train_sample*n_train_sample];
    chol_alpha = new float[n_train_sample*n_train_sample];

    flag_trained = inffunc -> compute_nlml(
                                            flag_grad, meta, x, y, kernel, meanfunc, likfunc, prior,
                                            chol_alpha, chol_factor_inv, beta, nlm_likelihood, dnlm_likelihood
                                            );
    if(!flag_trained){
        cout << "Warning: current inference failed in train()!!" << endl;
    }
}

vector <vector<float> > GP_Regression::predict(
                                                const vector<int> &meta, 
                                                const vector<int> &meta2, 
                                                const vector<float> &x, 
                                                const vector<float> &y, 
                                                const vector<float> &x2
                                                ){
    int i, j;
    int n_train_sample, n_test_sample;
    vector<float> pred_mean_vec;
    vector<float> pred_var_vec;
    vector< vector<float> > posterior;

    float *cross_gram_matrix, *cross_diag_matrix;
    float *mean_vec, *lik_vec;
    vector<float*> grad_mean, grad_lik; // unuse for now
    float *temp_vv_col, *pred_mean, *pred_var;

    // allocate space for arrays
    n_train_sample = int(y.size());
    n_test_sample = int(x2.size());

    assert(inf_thread_num >= 1);
    mkl_set_num_threads_local(inf_thread_num);
    omp_set_num_threads(inf_thread_num);

    cross_diag_matrix = new float[n_test_sample];
    cross_gram_matrix = new float[n_train_sample*n_test_sample];
    mean_vec = new float[n_test_sample];
    lik_vec = new float[n_test_sample];

    temp_vv_col = new float[n_train_sample];
    pred_mean = new float[n_test_sample];
    pred_var = new float[n_test_sample];

    // if the model is untrained, call the training function first
    if(!flag_trained){
        train(false, meta, x, y);
    }
    
    // compute the correlation of training and testing data
    meanfunc -> compute_mean_vector(meta2, x2, false, mean_vec, grad_mean);
    likfunc -> compute_lik_vector(meta2, x2, false, lik_vec, grad_lik);     
    kernel -> compute_cross_gram_matrix(meta, meta2, x, x2, cross_gram_matrix);
    kernel -> compute_self_diag_matrix(meta2, x2, cross_diag_matrix);
    
    // compute posterior
    MKL_INT n1 = n_train_sample;
    MKL_INT n2 = n_test_sample;
    MKL_INT lda = n_test_sample;
    
    // compute predictive mean
    cblas_scopy(n_test_sample, mean_vec, 1, pred_mean, 1);
    cblas_sgemv(CblasRowMajor, CblasTrans, n_train_sample, n_test_sample, 1.0f, cross_gram_matrix, lda, chol_alpha, 1, 1.0f, pred_mean, 1);
    
    // compute predictive variance
    // comput matrix v into cross_gram_matrix
    cblas_strmm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, n1, n2, 1.0f, chol_factor_inv, n1, cross_gram_matrix, n2);    
    for(i = 0; i < n_test_sample; i++){
        // copy the column out
        for(j = 0; j < n_train_sample; j++){
          temp_vv_col[j] = cross_gram_matrix[ j * n_test_sample + i];
        }
        pred_var[i] = cblas_sdsdot(n_train_sample, 0.0f, temp_vv_col, 1, temp_vv_col, 1);
    }

    for(i = 0; i < n_test_sample; i++){
        pred_var[i] = cross_diag_matrix[i] - pred_var[i] + lik_vec[i];
    }

    // copy the results to vector
    pred_mean_vec.assign(pred_mean, pred_mean+n_test_sample);
    pred_var_vec.assign(pred_var, pred_var+n_test_sample);
    posterior.push_back(pred_mean_vec);
    posterior.push_back(pred_var_vec);
    
    // free memory
    delete[] cross_gram_matrix;
    delete[] cross_diag_matrix;
    delete[] mean_vec;
    delete[] lik_vec;
    delete[] temp_vv_col;
    delete[] pred_mean;
    delete[] pred_var;

    return posterior;
}

vector <vector<float> > GP_Regression::parsed_predict(
                                                const int &meta_num,
                                                const vector<int> &meta, 
                                                const vector<int> &meta2, 
                                                const vector<float> &x, 
                                                const vector<float> &y, 
                                                const vector<float> &x2
                                                ){
    int i, j, k;
    int n_train_sample, n_test_sample;
    vector<float> pred_mean_vec;
    vector<float> pred_var_vec;
    vector< vector<float> > posterior;

    float *cross_gram_matrix, *cross_diag_matrix;
    float *mean_vec, *lik_vec;
    vector<float*> grad_mean, grad_lik; // unuse for now
    float *temp_vv_col, *pred_mean, *pred_var;

    // allocate space for arrays
    n_train_sample = int(y.size());
    n_test_sample = int(x2.size());

    assert(inf_thread_num >= 1);
    mkl_set_num_threads_local(inf_thread_num);
    omp_set_num_threads(inf_thread_num);

    cross_diag_matrix = new float[n_test_sample];
    cross_gram_matrix = new float[n_train_sample*n_test_sample];
    mean_vec = new float[n_test_sample];
    lik_vec = new float[n_test_sample];

    temp_vv_col = new float[n_train_sample];
    pred_mean = new float[n_test_sample];
    pred_var = new float[n_test_sample];

    // if the model is untrained, call the training function first
    if(!flag_trained){
        train(false, meta, x, y);
    }
    
    // compute the correlation of training and testing data
    meanfunc -> compute_mean_vector(meta2, x2, false, mean_vec, grad_mean);
    likfunc -> compute_lik_vector(meta2, x2, false, lik_vec, grad_lik);     
    kernel -> compute_cross_gram_matrix(meta, meta2, x, x2, cross_gram_matrix);
    kernel -> compute_self_diag_matrix(meta2, x2, cross_diag_matrix);
    // cout << "finish compute cross terms" << endl;
    
    // compute posterior
    MKL_INT n1 = n_train_sample;
    MKL_INT n2 = n_test_sample;
    MKL_INT lda = n_test_sample;
    
    // compute predictive mean
    cblas_scopy(n_test_sample, mean_vec, 1, pred_mean, 1);
    cblas_sgemv(CblasRowMajor, CblasTrans, n_train_sample, n_test_sample, 1.0f, cross_gram_matrix, lda, chol_alpha, 1, 1.0f, pred_mean, 1);
    
    // compute predictive variance
    // comput matrix v into cross_gram_matrix
    cblas_strmm(CblasRowMajor, CblasLeft, CblasLower, CblasNoTrans, CblasNonUnit, n1, n2, 1.0f, chol_factor_inv, n1, cross_gram_matrix, n2);    
    for(i = 0; i < n_test_sample; i++){
        // copy the column out
        for(j = 0; j < n_train_sample; j++){
          temp_vv_col[j] = cross_gram_matrix[ j * n_test_sample + i];
        }
        pred_var[i] = cblas_sdsdot(n_train_sample, 0.0f, temp_vv_col, 1, temp_vv_col, 1);
    }

    for(i = 0; i < n_test_sample; i++){
        pred_var[i] = cross_diag_matrix[i] - pred_var[i] + lik_vec[i];
    }

    // copy the results to vector
    pred_mean_vec.assign(pred_mean, pred_mean+n_test_sample);
    pred_var_vec.assign(pred_var, pred_var+n_test_sample);
    posterior.push_back(pred_mean_vec);
    posterior.push_back(pred_var_vec);
    
    // until above are the same, the following try tp parse based on the training sampels of each covariate separately
    kernel -> compute_cross_gram_matrix(meta, meta2, x, x2, cross_gram_matrix);
    for(i = 0; i < meta_num; i++){
        vector<float> decom_pred_mean_vec;
        cblas_scopy(n_test_sample, mean_vec, 1, pred_mean, 1);
        for(j = 0; j < n_test_sample; j++){
            for(k = 0; k < n_train_sample; k++){
                if(meta[k] == i){ // the target training covariate to decompose
                    pred_mean[j] += cross_gram_matrix[k * n_test_sample + j]*chol_alpha[k];
                }
            }
        }
        decom_pred_mean_vec.assign(pred_mean, pred_mean+n_test_sample);
        posterior.push_back(decom_pred_mean_vec);
    }

    // free memory
    delete[] cross_gram_matrix;
    delete[] cross_diag_matrix;
    delete[] mean_vec;
    delete[] lik_vec;
    delete[] temp_vv_col;
    delete[] pred_mean;
    delete[] pred_var;

    return posterior;
}
