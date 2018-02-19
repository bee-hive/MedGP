/*
-------------------------------------------------------------------------
This is the function file for single covariate Gaussian likelihood class.
-------------------------------------------------------------------------
*/
#include <iostream>
#include <math.h>
#include <vector>
#include <mkl.h>
#include <omp.h>
#include <time.h>
#include "likelihoods/c_likelihood_gaussianMO.h"
#include "util/global_settings.h"

using namespace std;


c_likelihood_gaussianMO::c_likelihood_gaussianMO(){
    likfunc_name = "c_likelihood_gaussianMO";
    likfunc_hyp_num = 1;
    cout << "WARNING: using multi-output likelihood function but no output number is specified!";
    cout << " Using default (1)" << endl;
}

c_likelihood_gaussianMO::c_likelihood_gaussianMO(vector<int> input_param)
                                                :c_likelihood(input_param){
    likfunc_name = "c_likelihood_gaussianMO";
    likfunc_hyp_num = input_param[0];
}

c_likelihood_gaussianMO::c_likelihood_gaussianMO(vector<int> input_param, vector<double> input_hyp)
                                                :c_likelihood(input_param, input_hyp){
    likfunc_name = "c_likelihood_gaussianMO";
    likfunc_hyp_num = input_param[0];

    if(int(input_hyp.size()) != likfunc_hyp_num){
        cout << "ERROR: mismatch # of hyperparameters! ";
        cout << "Get " << int(input_hyp.size()) << ", but expect " << likfunc_hyp_num << endl;
        exit(0);
    }
}

void c_likelihood_gaussianMO::compute_lik_vector(
                                                    const vector<int> &meta, 
                                                    const vector<float> &x, 
                                                    const bool &flag_grad, 
                                                    float *&lik_vector, 
                                                    vector<float*> &lik_gradients
                                                    ){
    int i, dim;
    dim = int(x.size());
    time_t  t1, t2;
    
    #pragma omp parallel for private(i) firstprivate(dim)
    for(i = 0; i < dim; i++){
        lik_vector[i] = pow(likfunc_hyp[meta[i]], 2.0);
    }

    if(flag_grad){
        // check if the input gradient vector is empty
        if(!lik_gradients.empty()){
            lik_gradients.clear();
        }
    }
}


