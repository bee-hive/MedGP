/*
-------------------------------------------------------------------------
This is the function file for the class exact inference.
-------------------------------------------------------------------------
*/
#include <iostream>
#include <vector>
#include <string>
#include <assert.h>
#include <mkl.h>
#include <omp.h>
#include <math.h>
#include <time.h>

#include "core/gp_model_include.h"
#include "inference/c_inference_exact.h"
#include "util/global_settings.h"

using namespace std;

c_inference_exact::c_inference_exact(){
    inffunc_name = "c_inference_exact";
    inf_thread_num = 1;
}
c_inference_exact::c_inference_exact(const int &thread_num){
    inffunc_name = "c_inference_exact";
    inf_thread_num = thread_num;
}
bool c_inference_exact::compute_nlml(
                                    const bool &flag_grad, 
                                    const vector<int> &meta, 
                                    const vector<float> &x, 
                                    const vector<float> &y, 
                                    c_kernel *kernel, 
                                    c_meanfunc *meanfunc, 
                                    c_likelihood *likfunc,
                                    c_prior *prior,
                                    float *&chol_alpha, 
                                    float *&chol_factor_inv, 
                                    float &beta,
                                    double &nlml, 
                                    vector<double> &dnlml
                                    ){
    // initialization
    bool    flag_success = false;
    char    *lower = "L";
    char    *nunit = "N";
    double  logDet = 0.0;
    double  quadDot = 0.0;
    int     i, j, k;
    int     count = 0;
    int     n_train_sample = int(y.size());

    // set local thread #
    assert(inf_thread_num >= 1);
    mkl_set_num_threads_local(inf_thread_num);
    omp_set_num_threads(inf_thread_num);

    // timing variables
    time_t  t1, t2, t3, t4;
    clock_t ct1, ct2;

    float *mean_vec, *lik_vec, *gram_matrix;
    vector<float*> grad_mean, grad_lik;

    mean_vec    = new float[n_train_sample];
    lik_vec     = new float[n_train_sample];
    gram_matrix = new float[n_train_sample*n_train_sample];

    // MKL parameters
    MKL_INT info;
    MKL_INT n = n_train_sample;
    MKL_INT lda = n_train_sample;

    time(&t3);

    // compute mean vector
    meanfunc -> compute_mean_vector(meta, x, flag_grad, mean_vec, grad_mean);
    cblas_sscal(n, -1, mean_vec, 1);
    cblas_saxpy(n, 1, &y[0], 1, mean_vec, 1);
    
    // compute likelihood vector
    likfunc -> compute_lik_vector(meta, x, flag_grad, lik_vec, grad_lik);

    // compute gram matrix
    time(&t1);
    kernel -> compute_self_gram_matrix(meta, x, gram_matrix);
    #pragma omp parallel for private(i) firstprivate(n_train_sample)
    for(i = 0; i < n_train_sample; i++){
        double temp_diag = gram_matrix[ i*n_train_sample + i ] + lik_vec[i];
        gram_matrix[ i*n_train_sample + i ] = (float)(temp_diag);
    }
    time(&t2);
    
    // do cholesky decomposition
    time(&t1);
    cblas_scopy(n*n, gram_matrix, 1, chol_factor_inv, 1);
    info = LAPACKE_spotrf(LAPACK_ROW_MAJOR, *lower, n, chol_factor_inv, lda);
    while((info != 0) && (count < 10)){
        cout << "WARNING: Cholesky decomposition failed! Start jittering count = " << count << endl;
        #pragma omp parallel for private(i) firstprivate(n_train_sample)
        for(i = 0; i < n_train_sample; i++){
            gram_matrix[ i*n_train_sample + i ] = gram_matrix[ i*n_train_sample + i ] + lik_vec[i];
        }
        cblas_scopy(n*n, gram_matrix, 1, chol_factor_inv, 1);
        info = LAPACKE_spotrf(LAPACK_ROW_MAJOR, *lower, n, chol_factor_inv, lda);
        count += 1; 
    }
    if(info != 0){
        return flag_success;
    }
    time(&t2);
    delete[] gram_matrix;

    // compute the log determinant
    flag_success = true;

    for(i = 0; i < n_train_sample; i++){
        logDet = logDet + log(chol_factor_inv[ i*n_train_sample + i ]);
    }

    // compute the inversion of gram matrix
    time(&t1);
    cblas_scopy(n, mean_vec, 1, chol_alpha, 1);
    info = LAPACKE_spotrs(LAPACK_ROW_MAJOR, *lower, n, 1, chol_factor_inv, n, chol_alpha, 1);
    time(&t2);  

    // compute the inverse of cholesky factor
    time(&t1);
    info = LAPACKE_strtri(LAPACK_ROW_MAJOR, *lower, *nunit, n, chol_factor_inv, lda);
    if(info != 0){
        cout << "ERROR: Cholesky factor inversion failed!" << endl;
        flag_success = false;
        return flag_success;
    }
    time(&t2);


    for(i = 0; i < (n_train_sample-1); i++){
        for(j = (i+1); j < n_train_sample; j++){
          chol_factor_inv[ i*n_train_sample + j ] = 0.0;
        }
    }

    // compute log marginal likelihood
    quadDot = cblas_dsdot(n, mean_vec, 1, chol_alpha, 1);
    beta = (float)(quadDot);

    double term1 = quadDot/2.0;
    double term2 = logDet;
    double term3 = n_train_sample*log(2.*PI)/2.0;
    nlml = term1 + term2 + term3;
    time(&t4);

    // compute gradients
    time(&t3);
    if(flag_grad){
        if(!dnlml.empty()){
            dnlml.clear();
        }

        float   *Q;
        double  sum = 0.0;
        int     offset = 0;
        vector<double> temp_hyp;

        // compute Q matrix
        Q = new float[n_train_sample*n_train_sample];
        
        time(&t1);
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, n, n, 1, 1.0f, chol_alpha, 1, chol_alpha, 1, 0.0f, Q, n);
        cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, n, n, n, 1.0f, chol_factor_inv, n, chol_factor_inv, n, -1.0f, Q, n);   
        time(&t2);

        // compute gradients for likelihood hypers
        time(&t1);
        temp_hyp = likfunc -> get_likfunc_hyp();
        if(meta.empty()){
            if(int(temp_hyp.size()) != 1){
                cout << "ERROR: the # of lik hyperparameter != 1 but no meta specified!" << endl;
                exit(1);
            }
            for(i = 0; i < int(temp_hyp.size()); i++){
                dnlml.push_back(0.0);
            }

            sum = 0.0;
            for(i = 0; i < n_train_sample; i++){
                sum = sum + pow(temp_hyp[0], 2.0)*Q[ i*n_train_sample + i];
            }
            dnlml[0] = sum;
        }
        else{
            for(i = 0; i < int(temp_hyp.size()); i++){
                dnlml.push_back(0.0);
                sum = 0.0;
                for(j = 0; j < n_train_sample; j++){
                    if(meta[j] == i)
                        sum = sum + pow(temp_hyp[i], 2.0)*(Q[ j*n_train_sample + j]);
                }
                dnlml[i] = sum;
            }           
        }
        offset += int(temp_hyp.size());
        time(&t2);

        // compute gradients for kernel hypers
        temp_hyp.clear();
        temp_hyp = kernel -> get_kernel_hyp();

        for(i = 0; i < int(temp_hyp.size()); i++){
            dnlml.push_back(0.0);
        }
        vector<double> grad_kernel_vec(temp_hyp.size(), 0.0);
        kernel -> compute_self_gradients(meta, x, Q, grad_kernel_vec);
        for(i = 0; i < int(temp_hyp.size()); i++){
            dnlml[ i + offset ] = grad_kernel_vec[i];
        }
        offset += int(temp_hyp.size());

        // compute gradients for mean hypers
        temp_hyp.clear();
        temp_hyp = meanfunc -> get_meanfunc_hyp();

        for(i = 0; i < int(temp_hyp.size()); i++){
            dnlml.push_back(0.0);
        }
        vector<double> grad_mean_vec(temp_hyp.size(), 0.0);
        meanfunc -> compute_mean_gradients(meta, x, chol_alpha, grad_mean_vec);
        for(i = 0; i < int(temp_hyp.size()); i++){
            dnlml[ i + offset ] = -1.*grad_mean_vec[i];
        }
        
        delete[] Q;
        
    }

    // free memory
    delete[] mean_vec;
    delete[] lik_vec;

    time(&t4);
    return flag_success;
}
