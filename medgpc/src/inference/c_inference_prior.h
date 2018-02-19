/*
-------------------------------------------------------------------------
This is the header file for the class prior inference.
-------------------------------------------------------------------------
*/
#ifndef C_INFERENCE_PRIOR_H
#define C_INFERENCE_PRIOR_H
#include <vector>
#include <string>

#include "kernel/c_kernel.h"
#include "mean/c_meanfunc.h"
#include "likelihoods/c_likelihood.h"
#include "inference/c_inference.h"
#include "prior/c_prior.h"

using namespace std;

class c_inference_prior:public c_inference{

    public:
        c_inference_prior();
        c_inference_prior(const int &thread_num);
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
