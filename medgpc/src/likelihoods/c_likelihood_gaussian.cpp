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
#include "likelihoods/c_likelihood_gaussian.h"
#include "util/global_settings.h"

using namespace std;


c_likelihood_gaussian::c_likelihood_gaussian(){
    likfunc_name = "c_likelihood_gaussian";
    likfunc_hyp_num = 1;
}
c_likelihood_gaussian::c_likelihood_gaussian(vector<int> input_param, vector<double> input_hyp)
                                            :c_likelihood(input_param, input_hyp){
    likfunc_name = "c_likelihood_gaussian";
    likfunc_hyp_num = 1;

    if(int(input_hyp.size()) != likfunc_hyp_num){
        cout << "ERROR: mismatch # of hyperparameters! ";
        cout << "Get " << int(input_hyp.size()) << ", but expect " << likfunc_hyp_num << endl;
        exit(0);
    }
}

void c_likelihood_gaussian::compute_lik_vector(
                                                const vector<int> &meta, 
                                                const vector<float> &x, 
                                                const bool &flag_grad, 
                                                float *&lik_vector, 
                                                vector<float*> &lik_gradients
                                                ){
    int i, dim;
    dim = int(x.size());
    time_t  t1, t2;

    // #pragma omp parallel for private(i) firstprivate(dim)
    for(i = 0; i < dim; i++){
        lik_vector[i] = pow(likfunc_hyp[0], 2.0);
    }

    if(flag_grad){
        // check if the input gradient vector is empty
        if(!lik_gradients.empty()){
            lik_gradients.clear();
        }
    }
}


