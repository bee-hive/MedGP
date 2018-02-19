/*
-------------------------------------------------------------------------
This is the function file for the class prior inference.
-------------------------------------------------------------------------
*/
#include <iostream>
#include <vector>
#include <string>
#include <math.h>
#include <time.h>
#include <assert.h>
#include "core/gp_model_include.h"
#include "util/global_settings.h"

using namespace std;

c_inference_prior::c_inference_prior(){
    inffunc_name = "c_inference_prior";
    inf_thread_num = 1;
}
c_inference_prior::c_inference_prior(const int &thread_num){
    inffunc_name = "c_inference_prior";
    inf_thread_num = thread_num;
}
bool c_inference_prior::compute_nlml(
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
    // get initial inference results from the major inference class
    bool flag_success;
    c_inference_exact   major_inffunc(inf_thread_num);
    flag_success = major_inffunc.compute_nlml(
                                                flag_grad, 
                                                meta, 
                                                x, 
                                                y, 
                                                kernel, 
                                                meanfunc, 
                                                likfunc,
                                                prior, 
                                                chol_alpha, 
                                                chol_factor_inv,
                                                beta, 
                                                nlml, 
                                                dnlml
                                                );

    if(flag_success && (prior != NULL)){        
        // timing variables
        time_t  t1, t2, t3, t4;
        int i, hyp_num, offset;
        vector<double> hyp;

        time(&t3);
        // adjust likelihood hyperparameters
        hyp = likfunc -> get_likfunc_hyp();// already hyperparameters after transformation
        hyp_num = int(hyp.size());
        
        offset = 0;
        for(i = 0; i < hyp_num; i++){
            // if active
            if(prior -> flag_lik[i]){
                // if clamped
                if((prior -> type_lik[i]) == 0){
                    if(flag_grad)
                        dnlml[i + offset] = 0.0;
                }
                else{
                    vector<double> lik;
                    lik = prior -> get_one_lik_lik(hyp[i], i);
                    nlml = nlml - lik[0];
                    if(flag_grad){
                        if(prior -> exp_lik[i])
                            dnlml[i + offset] = dnlml[i + offset] - hyp[i]*lik[1];
                        else
                            dnlml[i + offset] = dnlml[i + offset] - lik[1];
                    }
                }
            }
        }
        offset += hyp_num;

        // adjust covariance hyperparameters
        hyp = kernel -> get_kernel_hyp(); // already hyperparameters after transformation
        hyp_num = int(hyp.size());
        for(i = 0; i < hyp_num; i++){
            // if active
            if(prior -> flag_cov[i]){
                // if clamped
                if((prior -> type_cov[i]) == 0){
                    // force the gradient to be zero
                    if(flag_grad)
                        dnlml[i + offset] = 0.0;
                }
                else if((prior -> type_cov[i]) == -1){
                    // cout << "no prior" << endl; // do not change likelihood & gradient
                }
                else{
                    vector<double> lik;
                    lik = prior -> get_one_lik_cov(hyp[i], i);
                    nlml = nlml - lik[0];
                    if(flag_grad){
                        if(prior -> exp_cov[i])
                            dnlml[i + offset] = dnlml[i + offset] - hyp[i]*lik[1];
                        else
                            dnlml[i + offset] = dnlml[i + offset] - lik[1];
                    }
                }
            }
        }
        offset += hyp_num;

        // adjust mean hyperparameters
        hyp = meanfunc -> get_meanfunc_hyp(); // already hyperparameters after transformation
        hyp_num = int(hyp.size());
        // adjust mean function hyperparameters
        for(i = 0; i < hyp_num; i++){
            // if active
            if(prior -> flag_mean[i]){
                // if clamped
                if((prior -> type_mean[i]) == 0){
                    if(flag_grad)
                        dnlml[i + offset] = 0.0;
                }
                else{
                    vector<double> lik;
                    lik = prior -> get_one_lik_mean(hyp[i], i);
                    nlml = nlml - lik[0];
                    if(flag_grad){
                        if(prior -> exp_mean[i])
                            dnlml[i + offset] = dnlml[i + offset] - hyp[i]*lik[1];
                        else
                            dnlml[i + offset] = dnlml[i + offset] - lik[1];
                    }
                }
            }
        }

        time(&t4);
    }
    return flag_success;

}
