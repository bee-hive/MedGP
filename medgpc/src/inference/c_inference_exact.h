/*
-------------------------------------------------------------------------
This is the header file for the class exact inference.
Gaussian(MO) likelihood can be put into inference class directly.
-------------------------------------------------------------------------
*/
#ifndef C_INFERENCE_EXACT_H
#define C_INFERENCE_EXACT_H
#include <vector>
#include <string>

#include "kernel/c_kernel.h"
#include "mean/c_meanfunc.h"
#include "likelihoods/c_likelihood.h"
#include "inference/c_inference.h"
#include "prior/c_prior.h"

using namespace std;

class c_inference_exact:public c_inference{

    public:
        c_inference_exact();
        c_inference_exact(const int &thread_num);
        bool compute_nlml(
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
                            );

};
#endif